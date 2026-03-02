from openai import OpenAI

client = OpenAI(api_key="")

def jud_describe(vrp_text: str, problem_detail1_1: str, problem_detail2: str, model="gpt-4o-mini"):

    prompt = (
        "You are a VRP expert. I will give you:\n"
        "1) The original .vrp file content\n"
        "2) GPT's first answer (problem_description + constraints + specific_name)\n"
        "3) GPT's second answer (input, output, objective)\n\n"

        "Your task is to judge correctness:\n"
        "- For the first answer: check whether the problem_description, listed constraints, "
        "and specific_name are consistent with the VRP file TYPE and .vrp file contents. "
        "Specifically, check for contradictions in the following pairs:\n"
        "  • problem_description vs. .vrp file\n"
        "  • constraints vs. .vrp file\n"
        "  • specific_name vs. .vrp file\n"
        "  • problem_description vs. constraints\n"
        "  • problem_description vs. specific_name\n"
        "  • constraints vs. specific_name\n"
        "If any contradictions exist, treat the .vrp file as the ground truth and mark it as incorrect. "
        "If everything is consistent, mark it as correct.\n\n"
        "Check again the constraints whether fit specific name, and should have Visit constraint.\n"
        "If correct, return True with a short explanation. If wrong, return False with a short explanation.\n"

        "- For the second answer: check if input, output, and optimization objective are valid and consistent "
        "with the VRP instance and the constraints derived from the TYPE field. "
        "Input must correspond only to elements explicitly defined in the instance file "
        "(e.g., node_coordinates, demands, capacity, time_windows, distance_limits, fuel_capacity, fuel_consumption_rate, refuel_rate, stations),"
        "and don't consider vehicle_speed and map."
        "If the input includes time_windows or service_times, both must be present; if it includes capacity or demands, both must be present; and if it includes any of 'fuel_capacity, fuel_consumption_rate, refuel_rate, stations', then all of them must be present."
        "Every input listed must be something that can be directly obtained from the instance. "
        "Do not include any extra elements not present in the instance.\n"
        "- Input element names must not contain spaces; use underscores `_` instead.\n"
        "Output must clearly describe feasible vehicle routes respecting all constraints. "
        "The optimization objective must align with typical VRP goals (e.g., minimize total distance, minimize fleet size, minimize lateness).\n\n"
        "If any contradictions exist, treat the .vrp file as the ground truth and mark it as incorrect. "
        "If everything is consistent, mark it as correct.\n\n"
        "If correct, return True with a short explanation. If wrong, return False with a short explanation.\n"

        "Output format must be exactly 4 lines:\n"
        "1) right1: True/False\n"
        "2) jud1: explanation\n"
        "3) right2: True/False\n"
        "4) jud2: explanation\n\n"

        "Here is the VRP file:\n"
        f"{vrp_text}\n\n"

        "Here is GPT's first answer:\n"
        f"{problem_detail1_1}\n\n"

        "Here is GPT's second answer:\n"
        f"{problem_detail2}\n"
    )

    resp = client.responses.create(
        model=model,
        input=prompt
    )
    reply = resp.output_text.strip()
    lines = [line.strip() for line in reply.splitlines() if line.strip()]

    right1, jud1, right2, jud2 = False, "❌ Missing jud1", False, "❌ Missing jud2"

    if len(lines) >= 1 and "right1" in lines[0].lower():
        right1 = "true" in lines[0].lower()
    if len(lines) >= 2 and "jud1" in lines[1].lower():
        jud1 = lines[1].split(":", 1)[1].strip() if ":" in lines[1] else lines[1]

    if len(lines) >= 3 and "right2" in lines[2].lower():
        right2 = "true" in lines[2].lower()
    if len(lines) >= 4 and "jud2" in lines[3].lower():
        jud2 = lines[3].split(":", 1)[1].strip() if ":" in lines[3] else lines[3]

    return right1, jud1, right2, jud2


def jud_describe_previous(vrp_text: str, problem_detail1_1: str, model="gpt-4o-mini"):

    prompt = (
        "You are a VRP expert. I will give you:\n"
        "1) The original .vrp file content\n"
        "2) GPT's first answer (problem_description + constraints + specific_name)\n"

        "Your task is to judge correctness:\n"
        "- For the first answer: check whether the problem_description, listed constraints, "
        "and specific_name are consistent with the VRP file TYPE and .vrp file contents. "
        "Specifically, check for contradictions in the following pairs:\n"
        "  • problem_description vs. .vrp file\n"
        "  • constraints vs. .vrp file\n"
        "  • specific_name vs. .vrp file\n"
        "  • problem_description vs. constraints\n"
        "  • problem_description vs. specific_name\n"
        "  • constraints vs. specific_name\n"
        "If any contradictions exist, treat the .vrp file as the ground truth and mark it as incorrect. "
        "If everything is consistent, mark it as correct.\n\n"
        "If correct, return True with a short explanation. If wrong, return False with a short explanation.\n"

        "Output format must be exactly 2 lines:\n"
        "1) right1: True/False\n"
        "2) jud1: explanation\n"

        "Here is the VRP file:\n"
        f"{vrp_text}\n\n"

        "Here is GPT's first answer:\n"
        f"{problem_detail1_1}\n\n"
    )

    resp = client.responses.create(
        model=model,
        input=prompt
    )
    reply = resp.output_text.strip()
    lines = [line.strip() for line in reply.splitlines() if line.strip()]


    right1, jud1 = False, "❌ Missing jud1"

    if len(lines) >= 1 and "right1" in lines[0].lower():
        right1 = "true" in lines[0].lower()
    if len(lines) >= 2 and "jud1" in lines[1].lower():
        jud1 = lines[1].split(":", 1)[1].strip() if ":" in lines[1] else lines[1]

    return right1, jud1