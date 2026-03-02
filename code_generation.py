import os
import re
import argparse
from code_judgement import code_jud, code_overall_jud
from openai import OpenAI
client = OpenAI(api_key="")

def read_vrp_file(path: str, max_chars: int = 100000) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCATED: original file length={} chars]".format(len(text))
    return text

def build_prompt(vrp_path, vrp_text, problem_desc, constraints, specific_name, input_def, output_def, objective, func_name, prev_code=None) -> str:

    context = ""
    if prev_code:
        context = (
            "Here is the code you generated before (for reference, please improve or extend it if needed):\n"
            f"```\n{prev_code}\n```\n\n"
        )


    base_info = (
        f"We are working on a VRP problem instance from {vrp_path}.\n"
        f"Problem description: {problem_desc}\n"
        f"Constraints: {constraints}\n"
        f"Specific name: {specific_name}\n"
        f"Input definition: {input_def}\n"
        f"Output definition: {output_def}\n"
        f"Optimization objective: {objective}\n\n"
        "⚠️ Important rules:\n"
        "- You are given the raw .vrp file content below for context.\n"
        "- DO NOT hardcode any instance-specific details (e.g., number of nodes, vehicle count, node coordinates).\n"
        "- The functions you generate must be **general-purpose** and reusable for any VRP instance.\n\n"
        "Below is the .vrp instance content (for context only):\n"
        "```\n"
        f"{vrp_text}\n"
        "```\n\n"
    )

    task_map = {
    "read_vrp": (
        f"Generate a Python function named exactly `read_vrp(path: str)`.\n"
        f"- The function must read a `.vrp` file and parse its content strictly according to the items listed in Input definition: {input_def}.\n"
        f"- Every element mentioned in {input_def} must be parsed and included. No additional fields may be added, and no required fields may be omitted.\n"
        f"- The function must return a dictionary where:\n"
        f"  • Keys exactly match the names in {input_def}, using underscores `_` instead of spaces.\n"
        f"  • Each value must be returned as a list (array), even if it contains only a single element.\n"
        f"  A section may end with `-1`, `EOF`, or the beginning of another section header (e.g., lines in all caps ending with `_SECTION`), must handle all of these situations.\n"
        f"- If the output includes `demands`, `node_coordinates`, `service_times`, or `time_windows`, the corresponding depot information must also be included in those lists to maintain consistency."
        f"- Additionally, if 'Electricity' is in {constraints} and , the corresponding stations information must also be included in these lists.\n"
        f"- If the output includes 'stations' and the output includes `demands`, `node_coordinates`, `service_times`, or `time_windows`, all stations must be correctly listed, since stations are numerous.\n"
        f"- The implementation must be general-purpose. Do not hardcode any instance-specific details (e.g., node count, coordinates, capacity values).\n"
        f"- Output only the function code, enclosed in a single ```python ... ``` block."
    ),
    "distance": (
        "Generate a Python function named exactly `distance(coords)`.\n"
        "- This function must work seamlessly with the output of the previously defined `read_vrp(path: str)` function, "
        "where `coords` is the field containing node coordinates parsed from the .vrp file.\n"
        "- Input: `coords`, a list where each element is (x, y) representing the coordinates of a node. "
        "The index of each element corresponds to the node ID.\n"
        "- Output: a 2D list named `dist_matrix` such that:\n"
        "  • dist_matrix[i][j] = Euclidean distance between node i and node j.\n"
        "- Distances must be computed using the standard Euclidean formula: sqrt((x1 - x2)^2 + (y1 - y2)^2).\n"
        "- Return only this function definition. Do not include extra functions, comments, or explanations outside of it.\n"
    ),

    "cost": (
        "Generate a Python function named exactly `cost(route, dist_matrix)`.\n"
        "- Input:\n"
        "  • `route`: a list of node IDs (integers) representing the visiting order of nodes in the route.\n"
        "  • `dist_matrix`: a 2D list roduced by the previously defined `distance(coords)` function, "
        "where dist_matrix[i][j] gives the Euclidean distance between node i and node j.\n"
        "- The function must compute the total cost (distance) of the given route by summing the distances between consecutive nodes in the route, using values from `dist_matrix`.\n"
        #f"- If the problem constraint {constraints} contains 'Open Route', do not include the distance from the last node back to the depot at each route when computing the total cost.\n" 
        "- The function must return a value `total_cost` representing the total route distance.\n"
        "- Ensure indexing in `dist_matrix` is consistent with the node ID ordering used in `read_vrp` and `distance`.\n"
        "- Do not add extra functions, comments, or explanations outside the required function.\n"
    ),
    "initial": (
        "Generate a Python function named exactly `initial(instance, dist_matrix)`.\n"
        "- Purpose: Construct an initial feasible VRP solution using a greedy constructive method.\n"
        "- The solution must be built incrementally (node by node), adding customers step by step to routes in a greedy manner, always selecting the nearest feasible customer from the last visited node.\n"
        f"- Constraints that must be strictly respected during construction: {constraints}\n"
        "- Input: must use the parsed data `instance` from `read_vrp(path: str)` and the distance matrix `dist_matrix` from `distance(coords)`.\n"
        "- Output: return an initial solution structure (a list of routes, where each route is a list of node IDs including depot as required).\n"
        "- The function must ensure feasibility: no route violates any of the listed constraints.\n"
        "- When choosing next node, if several nodes are feasible, choose the one that is geographically nearest to the last visited node (greedy strategy based on distance using `dist_matrix`).\n"
        f"- Additionally, if {constraints} contains 'Open Route', then before selecting the next customer, compare the distance from the depot to its nearest remaining customer with the distance from the current last visited node to its nearest feasible customer; if the depot-nearest-customer distance is strictly smaller, terminate the current route and start a new route from the depot (open a new vehicle).\n"
        "- Each route must start at the depot and end at the depot unless the Open Route (O) constraint is active. "
        "When both Open Route (O) and Electricity (E) constraints are active, routes must not terminate at either the depot or a charging station (but vehicles may still visit charging stations during the route).\n"
        f"   • If {constraints} contains 'Electricity':\n"
        "       - Track the remaining fuel for each vehicle; at departure from depot, set remaining fuel to `fuel_capacity`.\n"
        "       - When evaluating a candidate customer, ensure that after traveling to that customer, the vehicle has enough remaining fuel to reach the nearest charging station (the nearest charging stationor the depot if closed routes) and then come back to depot.\n"
        "       - If 'Time Windows (TW)' are active, include both travel and recharging times in the feasibility check; service at a customer must complete within its time window.\n"
        "       - If fuel is not enough to go to other customer or depot, go to the nearst station."
        "       - If no feasible customer can be selected from the depot directly due to fuel or time, or cannot reach nearset station after reaching selected customer, explicitly attempt depot→station→customer sequences; select the nearest feasible combination that respects all constraints. This guarantees that initial customers can still be reached even if direct depot→customer travel is infeasible, while allowing at most one intermediate station visits.\n"
        "       - If no feasible customer can be selected as the next node from customer or station due to capacity, time, or distance limit, the vehicle go back to the depot if routes are closed( and can through at most one charging station if fuel is not enough to go back to depot respect to all constrains), and assign vehicle for new route.\n"
        "       - When visiting a charging station, compute recharge time as `(fuel_capacity - current_fuel) / refuel_rate` and update the current schedule.\n"
        "       - If the 'Open Route' constraint is active, and after visiting a charging station no further feasible customer can be served, discard the charging station from the end of the route and terminate at the last customer.\n"
        "       - Before come back to depot, check again the fuel is enough or not, if not, go to at most one station respect to all constrains.\n"
        "       - For node can not find the initial solution, must trys all combination of station in deopt->station->node->station(->depot if closed route) and all combination of station in deopt->node->station(->depot if closed route) which comfirm two station simultaneously instead one by one, and use feasible one as this node's initial route respect to all constrains, allowing at most one intermediate station visits.\n"
        "- The current time must always be updated as `current_time += travel_distance`, and additionally increased by waiting time (if arriving before time window) and by service/recharge times.\n"
        "- Ensure the generated code is consistent with `read_vrp`, `distance`, and `cost` functions.\n"
        "- Do not add extra helper functions or explanations outside the required function.\n"
    ),
    "destroy": (
        "Generate a Python function named exactly `destroy(instance, dist_matrix, solution, ratio)`.\n"
        "- Purpose: Remove a subset of nodes from the given solution based on the ratio.\n"
        "- The number of nodes to destroy must be computed as: `int(len(all_customers) * ratio)` where `all_customers` are all non-depot (and non-station nodes if Electricity (E) is active).\n"
        "- Destruction strategy:\n"
        "  • Step 1: Randomly choose one center node from the set of customers (must be non-depot (and non-station nodes if Electricity (E) is active)).\n"
        "  • Step 2: Using the distance matrix, build a candidate list of customer nodes ordered by increasing distance to the center; depots (and stations if Electricity (E) is active) are excluded; the center must be included at the head of this list.\n"
        "  • Step 3: Iterate through the candidate list in that order. For the current candidate (customer), locate the unique route that contains it; if that route has already been destroyed once, skip this candidate.\n"
        "  • Step 4: From this route, select a contiguous subsequence of nodes that includes the chosen customer.\n"
        "    - The subsequence length must be chosen randomly.\n"
        "    - The subsequence must include the chosen customer node.\n"
        "    - The subsequence must be fully contained within the route and must not cross depot boundaries.\n"
        "    - Depot nodes must not be included in the subsequence.\n"
        "    - If the Electricity (E) constraint is active, station nodes must also not be included in the subsequence (i.e., the subsequence contains only customers).\n"
        "    - Randomly pick one feasible subsequence (e.g., by sampling a valid length and start index) without exhaustively enumerating all candidates.\n"
        "  • Step 5: Remove the chosen contiguous subsequence from the route and append exactly those removed customer nodes to `removed_nodes` (never add nodes that were not actually deleted from the route). Then mark the route as destroyed and remove the chosen customer from the candidate list.\n"
        "  • Step 6: Repeat Steps 3-5 until the required number of customer nodes has been removed.\n"
        "  • A route cannot be destroyed more than once.\n"
        "  • If the Electricity (E) constraint is active, do not remove station nodes at any time. Stations may remain in the residual routes as intermediate stops.\n"
        "  • After destruction, clean up the residual routes:\n"
        "    - Merge any consecutive duplicate depot nodes into a single depot node.\n"
        "    - If the Electricity (E) and Open Route (O) constraints are both active, ensure no route terminates at a depot or a station: trim any trailing depots/stations so that each route ends at a customer; drop any route that becomes depot-only.\n"
        "- Final checks:\n"
        "  • Verify that every customer node appears in exactly one of: `removed_nodes` or the remaining `destroyed_solution` (customers are neither lost nor duplicated).\n"
        "  • Verify that no depot or station appears in `removed_nodes`.\n"
        "- Input:\n"
        "  • `instance`: the parsed dictionary returned by `read_vrp(path: str)`.\n"
        "  • `dist_matrix`: 2D list from `distance(coords)`.\n"
        "  • `solution`: the current solution as a list of routes (each route is a list of node IDs including depots/stations as present in the current solution format).\n"
        "  • `ratio`: a float (0 < ratio < 0.2) that determines the fraction of customer nodes to remove.\n"
        "- Output:\n"
        "  • `removed_nodes`: a list of removed customer node IDs (no depots/stations).\n"
        "  • `destroyed_solution`: the updated solution (list of routes) after destruction; its structure must be consistent with `initial`.\n"
        "- Constraints:\n"
        "  • Never destroy depots; if E is active, never destroy stations.\n"
        "  • Remove exactly `int(len(all_customers) * ratio)` customer nodes.\n"
        "  • Ensure outputs match the `solution` format (list of routes; each route a list of node IDs).\n"
        "- Ensure the generated code is consistent with `read_vrp`, `distance`, `cost`, and `initial`.\n"
        "- Do not add extra helper functions or explanations outside the required function.\n"
    ),

    "insert": (
        "Generate a Python function named exactly `insert(destroyed_solution, removed_nodes, instance, dist_matrix)`.\n"
        "- Purpose: Reinsert the previously removed nodes back into the solution.\n"
        "- Reinsertion strategy:\n"
        "  • Process nodes in the exact order they appear in `removed_nodes` (these are customers only; depots(and stations if Electricity (E) is active) must never appear here).\n"
        "  • For each node, evaluate all feasible insertion positions across all routes in `destroyed_solution`.\n"
        "  • For each candidate position, compute the incremental route cost using `dist_matrix` (indices consistent with `read_vrp` node order).\n"
        "  • Choose the insertion that yields the **minimal cost increase** while satisfying all constraints.\n"
        f"  • If the Open Route (O) constraint is active, may start a new route from the depot for this node **only if** it satisfies all constraints ({constraints}) and its cost increase is strictly smaller than every feasible insertion into existing routes.\n"
        f"  • If no feasible insertion exists in any route, create a new route for this node that satisfies all constraints ({constraints}).\n"
        "  • Ensure every node in `removed_nodes` is reinserted exactly once (no omissions, no duplicates).\n"
        "- Electricity-aware feasibility (if Electricity (E) is active):\n"
        "  • Treat `stations = instance['stations']` as charging-station nodes (not customers).\n"
        "  • Energy/fuel accounting:\n"
        "    - Let `fuel_capacity = instance['fuel_capacity'][0]`, `cons_rate = instance['fuel_consumption_rate'][0]`, `refuel_rate = instance['refuel_rate'][0]`.\n"
        "    - Travel of distance `d` consumes `d * cons_rate` units of fuel/energy.\n"
        "    - Charging at a station increases energy; assume **full recharge** to `fuel_capacity` with time penalty `(fuel_capacity - current_fuel) / refuel_rate` (assume stations have zero service time beyond recharge unless provided).\n"
        "  • Feasible insertion may insert **zero or more** charging stations near the insertion point (before and/or after the customer) to maintain energy feasibility between consecutive legs; depots must not be used as intermediate chargers unless closed routes are required."
        "  • If stations are inserted, the incremental route cost must include the extra travel via those station(s) and any induced recharge/service time\n"
        "  • When evaluating a candidate insertion, ensure that **every hop** along the modified route is energy-feasible (current_fuel never drops below 0), and if needed, insert the nearest reachable stations to bridge infeasible hops; account for their recharge time in time-window checks.\n"
        "  • When try to create a new route, must try deopt->station->node->station(->depot if closed route) and deopt->node->station(->depot if closed route), and choose the feasible one.\n"
        "  • If no feasible insertion exists in any route, create a new route for this node that satisfies all constraints ({constraints}).When try to create a new route, must try all conbination of station in deopt->station->node->station(->depot if closed route) and deopt->node->station(->depot if closed route) which comfirm two station simultaneously instead one by one, and choose the feasible one.\n"
        "  • Time windows and service times:\n"
        "    - Travel time equals distance by default; if `vehicle_speed` is provided in `instance`, use `time = distance / vehicle_speed`.\n"
        "    - Respect customer time windows and service times. For stations, add only recharge time (no demand) and respect any station time windows if provided.\n"
        "  • Open-route ending with E (O+E): after insertion, routes must **not** end at a depot or a station; if a route would end at a station (e.g., due to a final recharge), trim trailing stations so the route ends at the last customer. Drop depot-only or depot-station-only routes if they occur.\n"
        "- Constraints:\n"
        "  • Choose the insertion that yields the **minimal cost increase** instead of minimal cost while satisfying all constraints.\n"
        f"  • Every inserted node must satisfy all problem constraints specified in {constraints}.\n"
        "  • Depot nodes must never be inserted or removed during reinsertion; if if Electricity (E) is active, stations may only be inserted as support nodes for energy feasibility (not as customers).\n"
        "  • After each insertion, the solution must remain feasible.\n"
        "- Input:\n"
        "  • `destroyed_solution`: list of routes (each a list of node IDs) after destruction.\n"
        "  • `removed_nodes`: list of customer node IDs to reinsert (no depots/stations).\n"
        "  • `instance`: dictionary returned by `read_vrp(path: str)`.\n"
        "  • `dist_matrix`: 2D list returned by `distance(coords)`.\n"
        "- Output:\n"
        "  • A feasible `solution`: list of routes, where each route is a list of node IDs including depot(s) and, if needed, station nodes.\n"
        "- Post-processing & consistency:\n"
        "  • Merge any consecutive duplicate depot nodes that may arise.\n"
        "  • If O+E is active, ensure no route terminates at a depot or a station; ensure routes start at a depot.\n"
        "  • The implementation must be consistent with `read_vrp`, `distance`, `cost`, `initial`, and `destroy`.\n"
        "  • Cleanup: After all insertions, remove any route that serves no customers (i.e., routes containing only depot and/or station nodes).\n"
        "  • Station pruning (if Electricity (E) is active): iteratively try removing each charging-station stop from its route; If the route remains fully feasible and the total cost decreases, accept the removal and restart the scan. Repeat until no further cost-improving removals are possible.\n"
        "- Requirements:\n"
        "  • Maintain feasibility at all times; reinsert each removed node exactly once.\n"
        "  • Do not add extra helper functions, comments, or explanations outside the required function.\n"
    ),
    "validation": (
        "Generate a Python function named exactly `validate(solution, instance, dist_matrix)`.\n"
        "- Purpose: Validate whether a given VRP solution is feasible.\n"
        "- Input:\n"
        "  • `solution`: a list of routes, where each route is a list of node IDs (integers).\n"
        "  • `instance`: dictionary returned by `read_vrp(path: str)`.\n"
        "  • `dist_matrix`: 2D list returned by `distance(coords)` giving Euclidean distances between nodes.\n"
        "- Validation rules:\n"
        f"  • Respect all constraints explicitly listed in {constraints}.\n"
        "- Output:\n"
        "  • If the solution satisfies all constraints, return `True`.\n"
        "  • Otherwise, print the violated constraint(s) and return `False`.\n"
        "- Requirements:\n"
        "  • The function must check constraints one by one.\n"
        "  • It must be consistent with `read_vrp`, `distance`, `cost`, `initial`, `destroy`, and `insert`.\n"
        "  • Do not add extra helper functions, comments, or explanations outside the required function.\n"
    ),
    "main": (
        'Generate a Python function named exactly `if __name__ == "__main__":`.\n'
        "- Input arguments:\n"
        "  • Accept `path` (str) as a required input argument from command line, and must have `parser.add_argument('--path', type=str, help='Path to .vrp file')`\n"
        "  • Accept `iteration` (int) as an optional argument from command line, and must have `parser.add_argument('--iteration', type=int, default=100, help='Number of iterations (default: 100)')`.\n"
        "  • Use the `argparse` library to parse these arguments explicitly.\n"
        "  • The parsed values must be assigned to variables `path` and `iteration` for later use in the program.\n"
        "- Initialization:\n"
        "  • Initialize `best_solution = None` and `best_cost = float('inf')`.\n\n"
        "- Execution steps:\n"
        "  1. Call `read_vrp(path: str)` to parse the instance, assign to `instance`.\n"
        "  2. If no edge disatance matrix are given by instane, Extract coords from `instance['node_coordinates']` and call `distance(coords)` to get `dist_matrix`.\n"
        "  3. Call `initial(instance, dist_matrix)` to generate an initial solution, assign to `current_solution`.\n"
        "  4. Call `validate(solution=current_solution, instance=instance, dist_matrix=dist_matrix)`. "
        "     If it returns False, raise an Exception and exit.\n"
        "  5. Call `cost(route=current_solution, dist_matrix=dist_matrix)` to compute `current_cost`.\n"
        "     Assign `best_solution = current_solution` and `best_cost = current_cost`.\n"
        "     Print: `the initial process is successful, the initial cost is {best_cost}`.\n\n"
        "- Iterative improvement loop:\n"
        "  • For each step in range(iteration):\n"
        "    - Generate a random float `ratio` in (0, 0.2).\n"
        "    - Call `destroy(instance, dist_matrix, solution=current_solution, ratio=ratio)` → "
        "`removed_nodes, destroyed_solution`.\n"
        "    - Call `insert(destroyed_solution, removed_nodes, instance, dist_matrix)` → `current_solution`.\n"
        "    - Call `validate(solution=current_solution, instance=instance, dist_matrix=dist_matrix)`. "
        "If False, raise an Exception and exit.\n"
        "    - Call `cost(route=current_solution, dist_matrix=dist_matrix)` → `current_cost`.\n"
        "    - If `current_cost <=best_cost`, update `best_solution = current_solution`, `best_cost = current_cost`.\n"
        "    - Else (if `current_cost > best_cost`):\n"
        "         • Generate a random float `p` in (0,1).\n"
        "         • Compute threshold = exp(-(current_cost - best_cost) * iteration *10 / (iteration-step+1)).\n"
        "         • If `p > threshold`, then revert: `current_solution = best_solution`, `current_cost = best_cost`.\n\n"
        "- After loop ends:\n"
        "  • Print: `the process is successful, the best cost is {best_cost}`.\n\n"
        "- Constraints:\n"
        "  • Ensure the function uses only the functions `read_vrp`, `distance`, `initial`, `destroy`, `insert`, `validate`, and `cost` as described.\n"
        "  • Ensure consistent argument passing across all function calls.\n"
        "  • Do not add extra helper functions, comments, or explanations outside the required `main` function.\n"
    ),
}

    task = task_map.get(func_name, f"Generate a Python function for {func_name}.")


    return (
        context
        + base_info
        + task
        + "\n[Output format requirement: Return ONLY valid Python code, inside a single Python code block "
        "```python ... ``` with no extra text, explanations, or comments outside the code block.]"
        "If the code you generated before is provided, you must include it (or its improved/revised version) together with the newly generated function in the final output.]"
        "together with the newly generated function in the final output. "
        "Ensure that all necessary import statements required for the code to run are included at the top of the code block.]"
        )


def revise_prompt(vrp_path, vrp_text, problem_desc, constraints, specific_name, input_def, output_def, objective, prev_code, jud)-> str:
    context = ""
    if prev_code:
        context = (
            "Here is the code you generated before:\n"
            f"```\n{prev_code}\n```\n\n"
        )

    base_info = (
        f"We are working on a VRP problem instance from {vrp_path}.\n"
        f"Problem description: {problem_desc}\n"
        f"Constraints: {constraints}\n"
        f"Specific name: {specific_name}\n"
        f"Input definition: {input_def}\n"
        f"Output definition: {output_def}\n"
        f"Optimization objective: {objective}\n\n"
        "⚠️ Important rules:\n"
        "- You are given the raw .vrp file content below for context.\n"
        "- DO NOT hardcode any instance-specific details (e.g., number of nodes, vehicle count, node coordinates).\n"
        "- The functions you generate must be **general-purpose** and reusable for any VRP instance.\n\n"
        "Below is the .vrp instance content (for context only):\n"
        "```\n"
        f"{vrp_text}\n"
        "```\n\n"
    )
    return (
        context
        + base_info
        + "The code you generated previously has the following issues and revised suggestions:\n"
        f"{jud}\n\n"
        "Please correct the code according to the issues above, without changing the number of functions, their names, or their signatures.\n\n"
        "- If Time Windows is activate, don't consider depot (and charging staion if Electricity activate) time window in validation, but only customers' time window in validation.\n"
        f"In `read_vrp(path: str)`, every element mentioned in {input_def} must be parsed and included. No additional fields may be added, and no required fields may be omitted."
        "- If the output includes `demands`, `node_coordinates`, `service_times`, or `time_windows` in `read_vrp(path: str)`, the corresponding depot information must also be included in those lists to maintain consistency.\n"
        f"`read_vrp(path: str)` must return a dictionary where keys exactly match the names in {input_def}, using underscores `_` instead of spaces.\n\n"
        "[Output format requirement: Return ONLY valid Python code inside a single code block "
        "```python ... ``` with no extra text, explanations, or comments outside the code block.]\n"
    )

def extract_code(reply: str) -> str:

    match = re.search(r"```python(.*?)```", reply, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        print ("❌ No valid Python code block found in GPT reply.")
        return None

def ask_gpt(prompt: str, model: str = "gpt-4.1") -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        #max_output_tokens=512
    )
    return resp.output_text

def code_gen(vrp_path, problem_desc, constraints, specific_name, input_def, output_def, objective):
    vrp_text = read_vrp_file(vrp_path)
    #func_names=["read_vrp", "distance", "cost", "initial","destroy", "insert", "validation", "main"]
    out_dir = os.path.join(os.getcwd(), "code")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{specific_name}.py")
    func_names=["read_vrp", "distance", "cost", "initial","destroy", "insert", "validation", "main"]
    cost=[]
    prev_code = None
    task_id=0
    for func_name in func_names:
        task_id+=1
        cost_task=0
        right, jud = False, None
        while not right:
            cost_task+=1
            if jud is None:  
                prompt = build_prompt(vrp_path, vrp_text, problem_desc, constraints, specific_name, input_def, output_def, objective, func_name, prev_code=prev_code)
            else:  
                prompt = revise_prompt(vrp_path, vrp_text, problem_desc, constraints, specific_name, input_def, output_def, objective, prev_code, jud)
            raw_code = ask_gpt(prompt)
            #print("🔹 Raw code from GPT:\n", raw_code)
            code = extract_code(raw_code)
            prev_code = code  
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(code)
            right, jud = code_jud(vrp_path, vrp_text, problem_desc, constraints, specific_name, input_def, output_def, objective, code, task_id)
            #print(f"√ {func_name} Validation result: {right}, {jud}")
            prefix = "√" if right else "x"
            print(f"{prefix} {func_name} Validation result: {right}, {jud}")
        cost.append(cost_task)

    right, jud = False, None
    right, jud = code_overall_jud(vrp_path, vrp_text, problem_desc, constraints, specific_name, input_def, output_def, objective, prev_code)
    cost_task=0
    prefix = "√" if right else "x"
    print(f"{prefix} overall Validation result: {right}, {jud}")
    while right is False:
        cost_task+=1
        prompt = revise_prompt(vrp_path, vrp_text, problem_desc, constraints, specific_name, input_def, output_def, objective, prev_code, jud)
        raw_code = ask_gpt(prompt)
        code = extract_code(raw_code)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(code)
        right, jud = code_overall_jud(vrp_path, vrp_text, problem_desc, constraints, specific_name, input_def, output_def, objective, prev_code)
        prefix = "√" if right else "x"
        print(f"{prefix} overall Validation result: {right}, {jud}")
    cost.append(cost_task)
    sum_cost=sum(cost)
    print(f"Code for {specific_name} saved to {out_file}, cost: {cost}, sum_cost: {sum_cost}")
