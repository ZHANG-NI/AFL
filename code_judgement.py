from openai import OpenAI

client = OpenAI(api_key="")

def code_jud(vrp_path, vrp_text, problem_desc, constraints, specific_name, input_def, output_def, objective, code, task_id, model="gpt-4.1"):
    context = (
        "Here is the generated Python code:\n"
        f"```python\n{code}\n```\n\n"
    )
    func_names=["read_vrp", "distance", "cost", "initial","destroy", "insert", "validation", "main"]
    task_map = {
        "read_vrp":(
            "- For `read_vrp`:\n"
            "  • Ensure it extracts **exactly** the elements listed in the Input definition. \n"
            "  • No additional fields may be added, and no required fields may be omitted.\n"
            "  • Verify that each extracted element is explicitly available in the provided .vrp file content. \n"
            "  • Do not assume or fabricate values not present in the file.\n"
            f"  A section may end with `-1`, `EOF`, or the beginning of another section header (e.g., lines in all caps ending with `_SECTION`), must handle all of these situations.\n"
            "  • Confirm that the function returns all required fields in a dictionary format with each value must be returned as a list (array); "
            "    and keys must match the Input definition exactly, using underscores `_` instead of spaces.\n"
            "  • If the output includes `demands`, `node_coordinates`, `service_times`, or `time_windows`, the corresponding depot information must also be included in those lists to maintain consistency."
            f" • Additionally, if 'Electricity' is in {constraints} and , the corresponding stations information must also be included in these lists.\n"
            "  • If the output includes 'stations', verify that all stations are correctly listed, since stations are numerous.\n"
            "  • The function `read_vrp` must contained in the code.\n\n"
        ),
        "distance":(
            "- For `distance`:\n"
            "  • Confirm that the input argument `coords` has the correct format: "
            "  • A list where each element is (x, y) representing the coordinates of a node. \n"
            "  • Verify that the function returns a 2D list named `dist_matrix`, "
            "  • where dist_matrix[i][j] gives the Euclidean distance between node i and node j.\n"
            "  • Distances must be computed using the standard Euclidean formula: "
            "  • sqrt((x1 - x2)^2 + (y1 - y2)^2).\n"
            "  • The function `distance` must contained in the code.\n\n"
        ),
        "cost": (
            "- For `cost`:\n"
            "  • Input arguments:\n"
            "    - `route`: a list of node IDs (integers) representing the visiting order of nodes.\n"
            "    - `dist_matrix`: a 2D list (or NumPy array) produced by the previously defined `distance(coords)` function, where dist_matrix[i][j] gives the Euclidean distance between node i and node j.\n"
            "  • The function must compute the total cost (distance) of the given route by summing the distances between each pair of consecutive nodes in `route`, using values from `dist_matrix`.\n"
            #f"- If the problem constraint {constraints} contains 'Open Route', do not include the distance from the last node back to the depot at each route when computing the total cost.\n" 
            "  • The function must return a single numeric value `total_cost` representing the total distance of the route.\n"
            "  • Ensure that indexing into `dist_matrix` is consistent with the node ID ordering used in `read_vrp` and `distance`.\n"
            "  • The function `cost(route, dist_matrix)` must be present in the code.\n"
        ),
        "initial": (
            "- For `initial`:\n"
            "  • The function must be named exactly `initial(instance, dist_matrix)`.\n"
            "  • Input:\n"
            "    - `instance`: the parsed dictionary returned by `read_vrp(path: str)`.\n"
            "    - `dist_matrix`: the 2D list returned by `distance(coords)`.\n"
            "  • Output: an initial solution as a list of routes, where each route is a list of node IDs. "
            "  • Each route must start at the depot and end at the depot unless the Open Route (O) constraint is active. "
            "When both Open Route (O) and Electricity (E) constraints are active, routes must not terminate at either the depot or a charging station (but vehicles may still visit charging stations during the route).\n"
            f"  • Constraints: all constraints in {constraints} must be strictly enforced during route construction. "
            "  • Construction method: the solution must be built incrementally in a greedy step-by-step manner. "
            "    - When multiple feasible customers are available, the function must always choose the nearest one to the last visited node base `dist_matrix` unless the Open Route (O) constraint is active.\n"
            f"    - If {constraints} contains 'Open Route', then before selecting the next customer, compare the distance from the depot to its nearest remaining customer with the distance from the current last visited node to its nearest feasible customer; if the depot-nearest-customer distance is strictly smaller, terminate the current route and start a new route from the depot (open a new vehicle)."
            f"   • If {constraints} contains 'Electricity':\n"
            "       - Track the remaining fuel level for each vehicle, and begin from deopt, the remaining fuel must equal to 'fuel_capacity'.\n"
            "       - Ensure that when selecting the next customer, the vehicle must have enough remaining fuel not only to reach that customer, but also to subsequently reach the nearest charging station (nearest charging station or the depot if closed routes) from that customer's location, and then come back to depot.\n"
            "       - If fuel is not enough to go to other customer or depot, go to the one station respect to all constrains."
            "       - If 'Time Windows (TW)' are active, include the charging time into current time in feasibility checks to ensure no customer's window is violated.\n"
            "       - If no feasible customer can be selected from the depot directly due to fuel or time, or cannot reach nearset station after reaching selected customer, explicitly attempt depot→station→customer sequences respect to all constrains; select the nearest feasible combination that respects all constraints. This guarantees that initial customers can still be reached even if direct depot→customer travel is infeasible, while allowing at most one intermediate station visits.\n"
            "       - If no feasible customer can be selected as the next node from customer or station due to capacity, time, or distance limit, the vehicle go back to the depot if routes are closed( and can through at most one charging station if fuel is not enough to go back to depot respect to all constrains), and assign vehicle for new route.\n"
            "       - When visiting a charging station, must compute recharging time as `(fuel_capacity - current_fuel) / refuel_rate` and update the schedule.\n"
            "       - If the 'Open Route' constraint is active, and after visiting a charging station no further feasible customer can be served, the charging station should be discarded and the current route should terminate at the last customer.\n"
            "       - Before come back to depot, check again the fuel is enough or not, if not, go to at most one station respect to all constrains\n"
            "       - For node that can not find the initial solution, must use try all conbination of station in deopt->station->node->station(->depot if closed route) and deopt->node->station(->depot if closed route) which comfirm two station simultaneously instead one by one, and choose feasible one.\n"
            "  • The current time must can be increased by the same amount as the travel distance."
            "  • Consistency: the implementation must be consistent with the previously defined functions `read_vrp`, `distance`, and `cost`. "
            "  • The function must guarantee feasibility: no route may violate any of the constraints.\n"
            "  • The function `initial` must be present in the code.\n"
        ),
        "destroy":(
            "- For `destroy`:\n"
            "  • The function signature must be exactly `destroy(instance, dist_matrix, solution, ratio)`.\n"
            "  • Input:\n"
            "    - `instance`: dictionary returned by `read_vrp(path: str)`.\n"
            "    - `dist_matrix`: 2D list returned by `distance(coords)`.\n"
            "    - `solution`: list of routes, each route is a list of node IDs, including depots/stations as required.\n"
            "    - `ratio`: float (0 < ratio < 0.2).\n"
            "  • Customer set:\n"
            "    - Define `all_customers` as all non-depot nodes; if Electricity (E) is active, `all_customers` are all non-depot, non-station nodes.\n"
            "    - Compute the destruction quota as `int(len(all_customers) * ratio)`.\n"
            "  • Logic of destruction:\n"
            "    - Randomly choose one center node **from customers only** (never a depot; if E is active, never a station) and keep this single center for the whole destruction.\n"
            "    - Build a candidate list of customers ordered by **increasing** distance to the center using `dist_matrix` (center must be included at the head of the list; depots and, if E is active, stations are excluded).\n"
            "    - Iterate through the candidate list in that order. For the current candidate customer, locate the unique route that contains it; if that route has already been destroyed once, skip this candidate.\n"
            "    - From that route, select a contiguous subsequence that **includes the chosen customer** with a **randomly chosen length** subject to:\n"
            "      • The subsequence is fully contained within the route and does not cross depot boundaries.\n"
            "      • The subsequence must not contain any depot nodes.\n"
            "      • If E is active, the subsequence must not contain station nodes (customers only).\n"
            "      • Randomly pick one feasible subsequence (e.g., by sampling a valid length and start index) without exhaustively enumerating all candidates.\n"
            "    - Remove exactly the nodes in the chosen subsequence from the route; append **only those removed customer nodes** to `removed_nodes` (never add nodes that were not physically deleted from the route).\n"
            "    - Mark the route as destroyed (a route can be destroyed at most once). Remove the chosen neighbor from the candidate list and continue until the quota is reached.\n"
            "    - The total number of destroyed nodes must be **equal to** the computed quota (must not exceed it).\n"
            "  • Post-processing of `destroyed_solution`:\n"
            "    - Merge any consecutive duplicate depot nodes into a single depot node.\n"
            "    - If Open Route (O) and Electricity (E) are both active, ensure no route terminates at a depot or a station: trim any trailing depots/stations so each route ends at a customer; drop any route that becomes depot-only.\n"
            "  • Partition & integrity checks:\n"
            "    - Verify that every customer node appears in exactly one of: `removed_nodes` **or** the remaining `destroyed_solution` (no loss, no duplication).\n"
            "    - Verify `removed_nodes` contains no depots; if E is active, it must contain no stations.\n"
            "    - Verify all IDs in `removed_nodes` previously appeared in `solution` and do not appear in `destroyed_solution` after removal.\n"
            "    - Verify each route in `destroyed_solution` preserves the required format (list of node IDs in the same ID system used by `instance`).\n"
            "  • Consistency:\n"
            "    - Ensure the implementation is consistent with `read_vrp`, `distance`, `cost`, and `initial`.\n"
            "  • Output:\n"
            "    - Return `removed_nodes`: list of removed customer node IDs (unique, length equals quota, no depots; if E is active, no stations).\n"
            "    - Return `destroyed_solution`: updated solution (list of routes) consistent with the structure from `initial`.\n"
            "  • The function `destroy` must be present in the code.\n"
        ),
        "insert": (
            "- For `insert`:\n"
            "  • Confirm a function named exactly `insert(destroyed_solution, removed_nodes, instance, dist_matrix)` exists.\n\n"
            "  • Input checks:\n"
            "    - `destroyed_solution` is a list of routes; each route is a list of node IDs and starts at a depot (and may contain stations if E is active).\n"
            "    - `removed_nodes` is a list of customer node IDs only (no depots, no stations).\n"
            "    - `instance` matches the schema returned by `read_vrp` and all scalar fields are single-element lists.\n"
            "    - `dist_matrix` is a 2D list consistent with `distance(coords)` fuction.\n\n"
            "  • Reinsertion strategy compliance:\n"
            "    - Nodes must be processed strictly in the given `removed_nodes` order.\n"
            "    - For each node, the code evaluates all feasible insertion positions across all routes.\n"
            "    - The incremental route cost is computed via `dist_matrix`.\n"
            f"    - The chosen insertion must **minimizes the cost increase** while maintaining feasibility under {constraints}.\n"
            f"    - If the Open Route (O) constraint is active, starting a new route from the depot is allowed **only if** it respects all constraints ({constraints}) **and** its cost increase is strictly smaller than every feasible insertion into existing routes.\n"
            f"    - If no feasible insertion exists, the code creates a new feasible route for the node (respecting {constraints}).\n"
            "    - Every node in `removed_nodes` is reinserted exactly once (no omissions, no duplicates).\n\n"
            "  • Electricity-aware requirements (apply only if E is active):\n"
            "    - `stations = instance['stations']` are treated as charging nodes (not customers) and never appear in `removed_nodes`.\n"
            "    - Energy accounting uses: `fuel_capacity = instance['fuel_capacity'][0]`, `cons_rate = instance['fuel_consumption_rate'][0]`, `refuel_rate = instance['refuel_rate'][0]`.\n"
            "    - Each travel leg of length d consumes `d * cons_rate`; current fuel never drops below 0.\n"
            "    - The algorithm may insert zero or more stations **around the insertion point** (before and/or after the customer) to make all adjacent legs energy-feasible; depots must not be used as intermediate chargers unless closed routes are required.\n"
            "    - If stations are inserted, the cost increase must include the extra travel via the station(s) and any induced recharge/service time; recharge time is `(fuel_capacity - current_fuel) / refuel_rate`.\n"
            "    - Time windows and service times are respected at customers; for stations, only recharge time is applied. Travel time equals distance by default, or distance / vehicle_speed if provided.\n"
            "    - If no feasible insertion exists in any route, create a new route for this node that satisfies all constraints ({constraints}).\n"
            "    - When try to open new route, must try try all conbination of station in deopt->station->node->station(->depot if closed route) and deopt->node->station(->depot if closed route) which comfirm two station simultaneously instead one by one, and choose the feasible one.\n"
            "    - After insertion, if a route would end at a depot or station (O+E), trailing depots/stations are trimmed so the route ends at the last customer; routes with no customers are dropped.\n\n"
            "  • Solution-wide feasibility checks after all insertions:\n"
            f"    - All constraints in {constraints} hold.\n"
            "    - Choose the insertion that yields the **minimal cost increase** instead of minimal cost while satisfying all constraints.\n"
            "    - Depot nodes are never inserted/removed during reinsertion and never duplicated; stations, if present, appear only as support nodes for energy feasibility.\n"
            "    - Customers are unique across the whole solution (no customer appears twice).\n"
            "    - Merge any consecutive duplicate depots that may arise.\n"
            "    - **Cleanup:** remove any route that serves no customers.\n"
            "    - **Station pruning (if Electricity (E) is active):** attempt to remove recharging stations one by one, commit a removal only if the route remains fully feasible and the total cost strictly decreases. Repeat until no improving removal exists.\n\n"
            "  • Output checks:\n"
            "    - The function returns a feasible `solution` (list of routes) consistent with the formats produced by `initial` and `destroy`.\n"
            "    - Node IDs used in routes align with the `dist_matrix` indexing implied by `read_vrp`.\n\n"
            "  • Consistency checks:\n"
            "    - Implementation is consistent with `read_vrp`, `distance`, `cost`, `initial`, and `destroy`.\n"
            "    - No extraneous helper functions or outputs beyond the required `insert` definition.\n\n"
            ),
        "validation":(
            "- For `validation`:\n"
            "  • The function `validate(solution, instance, dist_matrix)` must be present in the code.\n"
            "  • Input:\n"
            "    - `solution`: must be a list of routes, each route a list of node IDs (integers).\n"
            "    - `instance`: must be the parsed dictionary returned by `read_vrp(path: str)`.\n"
            "    - `dist_matrix`: must be the 2D list returned by `distance(coords)`.\n"
            "  • Output:\n"
            "    - If all constraints are satisfied, the function must return `True`.\n"
            "    - If any constraint is violated, the function must print the violated constraint(s) and return `False`.\n"
            "  • Constraint validation:\n"
            f"    - The function must check all constraints explicitly listed in {constraints}, one by one.\n"
            "  • The implementation must be consistent with the `read_vrp`, `distance`, `cost`, `initial`, `destroy`, and `insert` function.\n"
            "  • The function `validation` must be present in the code.\n"
        ),
        "main":(
            '- For `if __name__ == "__main__":`:\n'
            '  • The function `if __name__ == "__main__"` must be present in the code.\n'
            "  • Accept `path` (str) as a required input argument from command line, and must have `parser.add_argument('--path', type=str, help='Path to .vrp file')`\n"
            "  • Accept `iteration` (int) as an optional argument from command line, and must have `parser.add_argument('--iteration', type=int, default=100, help='Number of iterations (default: 100)')`.\n"
            "  • The function must accept input arguments `path` (required) and `iteration` (optional, default=100) via argparse.\n"
            "  • Use the `argparse` library to parse these arguments explicitly.\n"
            "  • The parsed values must be assigned to variables `path` and `iteration` for later use in the program.\n"
            "  • The function must call the following in order:\n"
            "    1. `read_vrp(path: str)` → returns `instance`.\n"
            "    2. If no edge disatance matrix are given by instane, Extract coords from `instance['node_coordinates']` and call `distance(coords)` → `dist_matrix`.\n"
            "    3. Call `initial(instance, dist_matrix)` → `current_solution`.\n"
            "    4. Call `validate(solution=current_solution, instance=instance, dist_matrix=dist_matrix)` → must return True, otherwise raise an error.\n"
            "    5. Call `cost(route=current_solution, dist_matrix=dist_matrix)` → `current_cost`.\n"
            "    6. Initialize `best_solution = current_solution` and `best_cost = current_cost`, and Print: `the initial process is successful, the initial cost is {best_cost}`.\n"
            "  • Iterative loop (`for step in range(iteration)`):\n"
            "    - Generate a random float `ratio` in (0, 0.2).\n"
            "    - Call `destroy(instance, dist_matrix, solution=current_solution, ratio=ratio)` → `removed_nodes, destroyed_solution`.\n"
            "    - Call `insert(destroyed_solution, removed_nodes, instance, dist_matrix)` → `current_solution`.\n"
            "    - Call `validate(solution=current_solution, instance=instance, dist_matrix=dist_matrix)` → must return True.\n"
            "    - Call `cost(route=current_solution, dist_matrix=dist_matrix)` → `current_cost`.\n"
            "    - If `current_cost <= best_cost`, update `best_solution, best_cost`.\n"
            "    - If `current_cost > best_cost`, implement simulated annealing acceptance:\n"
            "        • Generate random float `p` in (0,1).\n"
            "        • Compute threshold = exp(-(current_cost - best_cost) * iteration *10 / (iteration-step+1)).\n"
            "        • If `p > threshold`, revert: `current_solution = best_solution`, `current_cost = best_cost`.\n"
            "  • After iterations, print: `the process is successful, the best cost is {best_cost}`.\n"
            "  • Ensure all steps are consistent with the signatures of `read_vrp`, `distance`, `initial`, `destroy`, `insert`, `validate`, and `cost`.\n"
            "  • Ensure all necessary imports (`argparse`, `random`, `math`) are included.\n"
        )

    }
    selected_rules = "".join(task_map.get(name, "") for name in func_names[:task_id])
    base_info = (
        f"We are working on a VRP problem instance from {vrp_path}.\n"
        f"Problem description: {problem_desc}\n"
        f"Constraints: {constraints}\n"
        f"Specific name: {specific_name}\n"
        f"Input definition: {input_def}\n"
        f"Output definition: {output_def}\n"
        f"Optimization objective: {objective}\n\n"
        f"Assume that this code is only work for {specific_name}, {problem_desc}. "
        "⚠️ Evaluation rules:\n"
        "- Only evaluate the given code snippet. Ignore any other functions or unrelated context.\n"
        "- Check if the code has syntax errors or logical bugs that would prevent execution.\n"
        + selected_rules
        + "Use the provided .vrp file content as the **only ground truth** for evaluation. "
        "Do not invent or assume data that is not present in the instance.\n\n"
        "Below is the .vrp instance content:\n"
        "```\n"
        f"{vrp_text}\n"
        "```\n\n"
    )

    prompt = (
        "You are a strict Python code reviewer and VRP expert.\n"
        + context
        + base_info
        + "Assume that in this VRP file format, whenever a section is present (e.g., NODE_COORD_SECTION, DEMAND_SECTION, TIME_WINDOW_SECTION, SERVICE_TIME_SECTION), it must contain entries for **all nodes, including the depot (and recharging station if 'Electricity' is activate in constrains)**, and no extra entries.\n. "
        f"Assume that the VRP file provide all element in {input_def}.\n. "
        f"Assume that this code is only work for {specific_name}, {problem_desc}. "
        "Assume that node IDs preserve the exact order given in the input .vrp file, are unique, and contain no duplicates."
        f"If 'insert' function exists, in 'insert' function, the chosen insertion must **minimizes the cost increase** while maintaining feasibility under {constraints}.\n"
        f"If 'initial' function exists, in 'initial' function, each step must respect to {constraints}.\n"
        "- If Time Windows is activate, don't consider depot (and charging staion if Electricity activate) time window in validation, but only customers' time window in validation.\n"
        + "Your task:\n"
        "- If the code is fully correct (no syntax errors, no logical bugs, all constraints satisfied, "
        "and fully consistent with the VRP rules), return `right1: True` and provide a brief explanation.\n"
        "- If the code has any issues (syntax bugs, logical errors, constraint violations, inconsistent naming, "
        "wrong input/output handling, or deviations from the specification), return `right1: False` and explain why. "
        "And if worng, you must also provide clear and concrete suggestions for how to fix or improve the code.\n\n"
        "⚠️ Important:\n"
        "Formatting rule: For easier parsing, the explanation or suggestions must be written in plain text on a single line, without using any line breaks (`\\n`) or additional colons (`:`) except the ones required in `right1:` and `jud1:`.\n\n"
        "- Output format must be exactly 2 lines:\n"
        "1) right1: True/False\n"
        "2) jud1: explanation and suggestions\n"
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


def code_overall_jud(vrp_path, vrp_text, problem_desc, constraints, specific_name, input_def, output_def, objective, prev_code, model="gpt-4.1"):
    context = (
        "Here is the generated Python code:\n"
        f"```python\n{prev_code}\n```\n\n"
    )
    base_info = (
        f"We are working on a VRP problem instance from {vrp_path}.\n"
        f"Problem description: {problem_desc}\n"
        f"Constraints: {constraints}\n"
        f"Specific name: {specific_name}\n"
        f"Input definition: {input_def}\n"
        f"Output definition: {output_def}\n"
        f"Optimization objective: {objective}\n\n"
        "⚠️ Evaluation rules:\n"
        f"If 'insert' function exists, in 'insert' function, the chosen insertion must **minimizes the cost increase** while maintaining feasibility under {constraints}.\n"
        f"If 'initial' function exists, in 'initial' function, each step must respect to {constraints}.\n"
        "- Only evaluate the given code snippet. Ignore any other functions or unrelated context.\n"
        "- Check if the code has syntax errors or logical bugs that would prevent execution.\n"
        "- Check if function calls between different parts of the code are consistent "
        "(e.g., argument names, parameter formats, return values).\n"
        #"- Verify that inputs/outputs align with the Input definition and Output definition.\n"
        "- Use the provided .vrp file content only as context for what data is available, "
        "but do not hardcode any instance-specific details.\n\n"
        "Below is the .vrp instance content:\n"
        "```\n"
        f"{vrp_text}\n"
        "```\n\n"
    )

    prompt = (
        "You are a strict Python code reviewer and VRP expert.\n"
        + context
        + base_info
        + "Assume that in this VRP file format, whenever a section is present (e.g., NODE_COORD_SECTION, DEMAND_SECTION, TIME_WINDOW_SECTION, SERVICE_TIME_SECTION), it must contain entries for **all nodes, including the depot**, and no extra entries.\n. "
        f"Assume that this code is only work for {specific_name}, {problem_desc}. "
        f"Assume that the VRP file provide all element in {input_def}.\n. "
        f"-If 'Electricity' activate, For node that can not find the initial solution in insert function, must try all combination of station in deopt->station->node->station(->depot if closed route) and deopt->node->station(->depot if closed route) which comfirm two station simultaneously instead one by one (station can be same or different), and choose feasible one respect to all constrains.\n"
        "- If 'Electricity' activate, If no feasible insertion exists in insert function, must create a new route for this node that satisfies all constraints. When try to create a new route, must try try all combination of station in deopt->station->node->station(->depot if closed route) and deopt->node->station(->depot if closed route) which comfirm two station simultaneously instead one by one, and choose the feasible one(station can be same or different).\n"
        + "Your task:\n"
        "- If you find any issues (bugs, wrong parameter passing, inconsistencies), return `right1: False`, explain why is wrong and provide clear and concrete suggestions for how to fix or improve the code.\n"
        "- If the code is fully correct and no improvements are needed, return `right1: True` and explain briefly why it is correct.\n"
        "⚠️ Important formatting rule: For easier parsing, the explanation or suggestions must be written in plain text on a single line, without using any line breaks (`\\n`) or additional colons (`:`) except the ones required in `right1:` and `jud1:`.\n\n"
        "Output format must be exactly 2 lines:\n"
        "1) right1: True/False\n"
        "2) jud1: explanation or suggestions\n"
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
