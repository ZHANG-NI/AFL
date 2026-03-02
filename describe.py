import os
import re
import argparse
from openai import OpenAI
from describe_judgement import jud_describe, jud_describe_previous
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api2d.net/v1")
client = OpenAI(api_key="")

def read_vrp_file(path: str, max_chars: int = 100000) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCATED: original file length={} chars]".format(len(text))
    return text

def build_prompt_part1(vrp_text: str, jud:str, ans:str) -> str:

    if jud is None:
        return (
            "We need to solve a VRP instance. I will provide you with the instance. "
            "Please analyze it carefully. "
            "First, give a concise description of the problem type in [ ], explaining what the problem is about, and 'E' represents Electricity. " 
            "Second, identify its constraints and list them clearly in numbered format (1), 2), 3), ...) within [ ]. " 
            "For each constraint, write both the abbreviation (if any) and a short explanation " 
            "(e.g., 'Capacity (C): vehicles have limited capacity.'). " 
            "Do not include instance-specific details like the exact number of nodes, vehicles, or capacity values. " 
            "Important rules for constraints:\n" 
            "- Only include 'L' (Duration Limit) if the instance explicitly specifies a maximum route distance or duration limit. " 
            "Do NOT confuse time windows with distance limits.\n" 
            "- Only include 'O' (Open Route) if the instance explicitly specifies 'O' in the 'TYPE'.\n" 
            "- Only include 'E' (Electricity) if the instance explicitly specifies 'FUEL_CAPACITY' which is related to Electricity.\n"
            "When analyzing, be as comprehensive as possible: " 
            "consider not only the common constraints but also more general ones, such as whether each customer can be visited multiple times, " 
            "whether all routes must start and end at a depot, or other structural constraints that might apply. " 
            "You may refer to the following typical constraint categories as guidance, but you are not limited to them:\n" 
            "- Electricity (E): electric vehicles are subject to fuel constraints. "
            "Each vehicle has a limited 'FUEL_CAPACITY', fuel is consumed proportionally to the distance traveled that is relate to 'FUEL_CONSUMPTION_RATE', and vehicles must recharge at designated charging stations when necessary. "
            "Recharging operations consume time that related to the 'REFUEL_RATE' and current fuel remained, which must also be considered when planning routes with 'Time Windows (TW)' constrain.\n"
            "- Capacity (C): vehicles have limited capacity.\n" 
            "- Open Route (O): vehicles do not return to the depot.\n" 
            "- Backhaul (B): A routing setting where vehicles first execute deliveries (linehaul phase) and then perform pickups (backhaul phase) before returning to the depot. During the linehaul phase, the vehicle departs the depot fully loaded with goods for delivery, and the cumulative amount of goods remaining on board at any point must not exceed the vehicle capacity. "
            "Once all deliveries are completed, the backhaul phase begins. In this phase, the vehicle starts from an empty state (i.e., zero load), and the cumulative quantity of collected pickups at any point must not exceed the vehicle capacity.\n"
            "- Mixture Backhaul (MB): Deliveries (linehaul phase) and pickups (backhaul phase) are mixed within a single route, allowing interleaved operations. The Mixture Backhauls constraint requires that both the total delivered (linehaul) load and the total collected (backhaul) load within a route must individually not exceed the vehicle capacity. "
            "Additionally, The Mixture Backhauls requires that, at every customer visit along a route, the sum of the total quantity of goods already picked up (i.e., the cumulative amount of all previous backhaul pickups) and the absolute demand of the current customer must not exceed the vehicle’s capacity. This condition ensures that, at any point in the route, the combination of previously collected backhaul loads and the current customer’s delivery or pickup demand never causes the vehicle to exceed its maximum load capacity.\n"
            "- Distance Limit (L): each route has a maximum distance or time limit.\n"
            "- Time Windows (TW): customers must be served within specific time intervals.\n"
            "- Multi-depot (MD): multiple depots instead of one.\n"
            "- Visit constraint (V): each customer may only be visited once (or specify if multiple visits are allowed).\n" 
            "- Depot constraint (D): routes must start and end at the depot (unless open routes are specified).\n" 
            "Finally, write the standard problem type abbreviation (e.g., TSP, CVRP, CVRPL, VRPTW, PDP, OVRP, MDVRP, ECVRP) " 
            "enclosed in \" \". \n" 
            "Check again the constraints whether fit specific name, and should have Visit constraint.\n"#, such as 'Open Route (O)','Duration Limit (L)' and so on.\n"
            "Output exactly three lines in this order:\n"
            "1) [problem description]\n"
            "2) [constraints]\n"
            "3) \"specific name\"\n\n"
            "Below is the .vrp instance content:\n"
            + vrp_text
            + "\n```"
        )
    else:
        return (
            "We need to solve a VRP instance. I will provide you with the instance. "
            "Please analyze it carefully.\n\n"
            "Step 1: Give a concise description of the problem type in [ ], explaining what the problem is about.\n"
            "Step 2: Identify its constraints and list them clearly in numbered format (1), 2), 3), ...) within [ ]. "
            "For each constraint, write both the abbreviation (if any) and a short explanation "
            "(e.g., 'Capacity (C): vehicles have limited capacity.').\n"
            "Do not include instance-specific details like the exact number of nodes, vehicles, or capacity values.\n\n"

            "Important rules for constraints:\n"
            "- Only include 'L' (Duration Limit) if the instance explicitly specifies a maximum route distance or duration limit. "
            "Do NOT confuse time windows with distance limits.\n"
            "- Include 'O' (Open Route) if the instance explicitly specifies 'O' in the 'TYPE'.\n"
            "- Only include 'E' (Electricity) if the instance explicitly specifies 'FUEL_CAPACITY' which is related to Electricity.\n"
            "- Be comprehensive: consider also if customers may be visited once or multiple times, "
            "and whether all routes must start/end at a depot (unless open routes are specified).\n"
            "Reference constraint categories (not exhaustive):\n"
            "- Electricity (E): electric vehicles are subject to fuel constraints. "
            "Each vehicle has a limited 'FUEL_CAPACITY', fuel is consumed proportionally to the distance traveled that is relate to 'FUEL_CONSUMPTION_RATE', and vehicles must recharge at designated charging stations when necessary. "
            "Recharging operations consume time that related to the 'REFUEL_RATE' and current fuel remained, which must also be considered when planning routes with 'Time Windows (TW)' constrain.\n"
            "- Capacity (C): vehicles have limited capacity.\n"
            "- Open Route (O): vehicles do not return to the depot.\n"
            "- Backhaul (B): A routing setting where vehicles first execute deliveries (linehaul phase) and then perform pickups (backhaul phase) before returning to the depot. During the linehaul phase, the vehicle departs the depot fully loaded with goods for delivery, and the cumulative amount of goods remaining on board at any point must not exceed the vehicle capacity. "
            "Once all deliveries are completed, the backhaul phase begins. In this phase, the vehicle starts from an empty state (i.e., zero load), and the cumulative quantity of collected pickups at any point must not exceed the vehicle capacity.\n"
            "- Mixture Backhaul (MB): Deliveries (linehaul phase) and pickups (backhaul phase) are mixed within a single route, allowing interleaved operations. The Mixture Backhauls constraint requires that both the total delivered (linehaul) load and the total collected (backhaul) load within a route must individually not exceed the vehicle capacity. "
            "Additionally, The Mixture Backhauls requires that, at every customer visit along a route, the sum of the total quantity of goods already picked up (i.e., the cumulative amount of all previous backhaul pickups) and the absolute demand of the current customer must not exceed the vehicle’s capacity. This condition ensures that, at any point in the route, the combination of previously collected backhaul loads and the current customer’s delivery or pickup demand never causes the vehicle to exceed its maximum load capacity.\n"
            "- Distance Limit (L): each route has a maximum distance or time limit.\n"
            "- Time Windows (TW): customers must be served within specific time intervals.\n"
            "- Multi-depot (MD): multiple depots instead of one.\n"
            "- Visit constraint (V): whether each customer can be visited only once.\n"
            "- Depot constraint (D): routes must start and end at the depot (unless open routes).\n\n"

            "Step 3: Write the standard problem type abbreviation (e.g., TSP, CVRP, CVRPL, VRPTW, PDP, OVRP, MDVRP, ECVRP) "
            "enclosed in \" \". Ensure the abbreviation is consistent with the constraints.\n\n"
            "Check again the constraints whether fit specific name, and should have Visit constraint.\n"#, such as 'Open Route (O)','Duration Limit (L)' and so on.\n"
            "Here is your previous answer:\n"
            + ans + "\n\n"
            "However, there were some issues identified:\n"
            + jud + "\n\n"
            "Now, please correct your answer strictly according to the rules above.\n\n"

            "Output format (exactly three lines):\n"
            "1) [problem description]\n"
            "2) [constraints]\n"
            "3) \"specific name\"\n\n"
            "Below is the .vrp instance content:\n"
            + vrp_text
            + "\n```"
        )


def build_prompt_part2(vrp_text: str, problem_detail: str, jud:str, ans:str) -> str:

    if jud is None:
        return (
            "We need to design an algorithm for the following VRP instance. "
            f"The details of the instance are: {problem_detail}. "
            "Based on this description and the instance contents, please specify:\n"
            "First, list the essential elements an algorithm would require from the instance "
            "(e.g., depot, node_coordinates, demands, capacity, service_times, time_windows, distance_limits, fuel_capacity, fuel_consumption_rate, refuel_rate, stations if present),"
            "don't consider vehicle_speed and map."
            "If the input includes time_windows or service_times, both must be present; if it includes capacity or demands, both must be present; and if it includes any of 'fuel_capacity, fuel_consumption_rate, refuel_rate, stations', then all of them must be present.\n"
            "Second, describe precisely what the algorithm should output "
            "(e.g., a best feasible vehicle route that satisfy all listed constraints).\n"
            "Third, describe clearly the optimization objective "
            "(e.g., minimize total travel distance, minimize fleet size, minimize lateness).\n\n"
            "Important: Each of the three answers (input, output, objective) must be enclosed in [ ] as shown.\n\n"
            "Do not include instance-specific details like the exact number of nodes, vehicles, or capacity values. " 
            "And check again the input are all provided in instance content"
            "- First step element names must not contain spaces; use underscores `_` instead.\n"
            "Output exactly three lines in this order:\n"
            "4) []\n"
            "5) []\n"
            "6) []\n\n"
            "Below is the .vrp instance content:\n"
            + vrp_text
            + "\n```"
        )
    else:
        return (
            "We need to design an algorithm for the following VRP instance. "
            f"The details of the instance are: {problem_detail}. "
            "Based on this description and the instance contents, please provide:\n\n"

            "Step 1: List the essential elements an algorithm would require from the instance, must include depot"
            "(e.g., depot, node_coordinates, demands, capacity, service_times, time_windows, distance_limits, fuel_capacity, fuel_consumption_rate, refuel_rate, stations if present).\n"
            ", don't consider vehicle_speed and map."
            "If the input includes time_windows or service_times, both must be present; if it includes capacity or demands, both must be present; and if it includes any of 'fuel_capacity, fuel_consumption_rate, refuel_rate, stations', then all of them must be present.\n"
            "Step 2: Describe precisely what the algorithm should output "
            "(e.g., a set of feasible vehicle routes that satisfy all listed constraints).\n"
            "Step 3: Describe clearly the optimization objective "
            "(e.g., minimize total travel distance, minimize fleet size, minimize lateness).\n\n"

            "Important rules:\n"
            "- Each of the three answers (input, output, objective) must be enclosed in [ ] exactly as shown.\n"
            "- Do not include instance-specific details like the exact number of nodes, vehicles, or capacity values.\n\n"
            "- Step 1 element names must not contain spaces; use underscores `_` instead.\n"
            "Here is your previous answer:\n"
            + ans + "\n\n"
            "Issues identified in that answer:\n"
            + jud + "\n\n"
            "Now, please correct your answer strictly according to the rules above.\n\n"

            "Final output format (exactly three lines):\n"
            "4) []\n"
            "5) []\n"
            "6) []\n\n"

            "Below is the .vrp instance content:\n"
            + vrp_text
            + "\n```"
        )


def extract_part1(text: str):

    quote_pattern = r'["“”„‟«»「」『』＂]'
    desc_matches = re.findall(r"\[(.+?)\]", text, flags=re.DOTALL)

    problem_description = desc_matches[0].strip() if len(desc_matches) >= 1 else None
    constraints = desc_matches[1].strip() if len(desc_matches) >= 2 else None

    # 找 specific name
    name_match = re.search(fr"{quote_pattern}(.+?){quote_pattern}", text, flags=re.DOTALL)
    specific_name = name_match.group(1).strip() if name_match else None

    def clean(s): return " ".join(s.split()) if s else None
    return clean(problem_description), clean(constraints), clean(specific_name)


def extract_part2(text: str):

    desc_matches = re.findall(r"\[(.+?)\]", text, flags=re.DOTALL)
    input_def = desc_matches[0].strip() if len(desc_matches) >= 1 else None
    output_def = desc_matches[1].strip() if len(desc_matches) >= 2 else None
    objective = desc_matches[2].strip() if len(desc_matches) >= 3 else None

    def clean(s): return " ".join(s.split()) if s else None
    return clean(input_def), clean(output_def), clean(objective)

def ask_gpt(prompt: str, model: str = "gpt-4.1-mini") -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        #max_output_tokens=512
    )
    return resp.output_text

def describe_vrp(vrp_path, model="gpt-4.1", max_chars=100000, vrp_list=[]):

    jud1, jud2 = None, None
    ans1, ans2 = None, None
    right1, right2 = False, False
    i=0
    input_def, output_def, objective=None, None, None
    while right1 ==False or right2 ==False:
        i+=1
        try:
            vrp_text = read_vrp_file(vrp_path, max_chars=max_chars)
        except Exception as e:
            raise RuntimeError(f"Failed to read VRP file {vrp_path}: {e}")

        prompt1 = build_prompt_part1(vrp_text, jud1, ans1)
        try:
            ans1 = ask_gpt(prompt1, model=model)
            # print("RAW reply (part1):\n", reply1)
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed (part1): {e}")

        problem_desc, constraints, specific_name = extract_part1(ans1)
        problem_detail1 = f"problem_description: {problem_desc} \nconstraints: {constraints}\n"
        problem_detail1_1 = f"problem_description: {problem_desc} \n specific type: {specific_name} \n constraints: {constraints}\n"
        if specific_name in vrp_list:
            right1, jud1 = jud_describe_previous(vrp_text, problem_detail1_1)
            right2=True
            continue

        prompt2 = build_prompt_part2(vrp_text, problem_detail1, jud2, ans2)
        try:
            ans2 = ask_gpt(prompt2, model=model)
            # print("RAW reply (part2):\n", reply2)
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed (part2): {e}")

        input_def, output_def, objective = extract_part2(ans2)
        problem_detail2 = f"input_definition: {input_def} \n output_definition: {output_def}\n optimization: {objective}"
        right1, jud1, right2, jud2 = jud_describe(vrp_text, problem_detail1_1, problem_detail2)
        print(i, right1, jud1, "\n", right2, jud2)
    #print(i)
    if specific_name not in vrp_list:
        vrp_list.append(specific_name)
    return i, problem_desc, constraints, specific_name, input_def, output_def, objective, vrp_list



if __name__ == "__main__":
    vrp_path="./vrp/cvrp/50/838.vrp"
    i, problem_desc, constraints, specific_name, input_def, output_def, objective, _= describe_vrp(vrp_path)
    print("time", i)
    print("problem_description:", problem_desc or "❌ Not found")
    print("constraints:", constraints or "❌ Not found")
    print("specific_name:", specific_name or "❌ Not found")
    print("input:", input_def or "❌ Not found")
    print("output:", output_def or "❌ Not found")
    print("objective:", objective or "❌ Not found")