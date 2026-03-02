import math
import random
import argparse
import sys

def read_vrp(path: str):
    depot = None
    node_coordinates = []
    demands = []
    service_times = []
    time_windows = []
    vehicle_capacity = None
    dimension = None

    required_sections = {
        "NODE_COORD_SECTION": False,
        "DEMAND_SECTION": False,
        "TIME_WINDOW_SECTION": False,
        "SERVICE_TIME_SECTION": False,
        "DEPOT_SECTION": False
    }

    with open(path, 'r') as f:
        lines = f.read().splitlines()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith('CAPACITY'):
            try:
                vehicle_capacity = float(line.split(':')[1].strip())
            except Exception:
                raise ValueError("Could not parse CAPACITY field.")
        elif line.startswith('DIMENSION'):
            try:
                dimension = int(line.split(':')[1].strip())
            except Exception:
                raise ValueError("Could not parse DIMENSION field.")
        elif line.startswith('NODE_COORD_SECTION'):
            required_sections["NODE_COORD_SECTION"] = True
            i += 1
            while i < n:
                cline = lines[i].strip()
                if cline == '' or cline.startswith('DEMAND_SECTION'):
                    break
                tokens = cline.split()
                if len(tokens) < 3:
                    i += 1
                    continue
                coords = [float(tokens[1]), float(tokens[2])]
                node_coordinates.append(coords)
                i += 1
            continue
        elif line.startswith('DEMAND_SECTION'):
            required_sections["DEMAND_SECTION"] = True
            i += 1
            while i < n:
                dline = lines[i].strip()
                if dline == '' or dline == '-1' or dline.startswith('TIME_WINDOW_SECTION'):
                    break
                tokens = dline.split()
                if len(tokens) < 2:
                    i += 1
                    continue
                demand = float(tokens[1])
                demands.append(demand)
                i += 1
            continue
        elif line.startswith('TIME_WINDOW_SECTION'):
            required_sections["TIME_WINDOW_SECTION"] = True
            i += 1
            while i < n:
                twline = lines[i].strip()
                if twline == '' or twline == '-1' or twline.startswith('SERVICE_TIME_SECTION'):
                    break
                tokens = twline.split()
                if len(tokens) < 3:
                    i += 1
                    continue
                tw1, tw2 = float(tokens[1]), float(tokens[2])
                time_windows.append([tw1, tw2])
                i += 1
            continue
        elif line.startswith('SERVICE_TIME_SECTION'):
            required_sections["SERVICE_TIME_SECTION"] = True
            i += 1
            while i < n:
                stline = lines[i].strip()
                if stline == '' or stline == '-1' or stline.startswith('DEPOT_SECTION'):
                    break
                tokens = stline.split()
                if len(tokens) < 2:
                    i += 1
                    continue
                stime = float(tokens[1])
                service_times.append(stime)
                i += 1
            continue
        elif line.startswith('DEPOT_SECTION'):
            required_sections["DEPOT_SECTION"] = True
            i += 1
            while i < n:
                dpline = lines[i].strip()
                if dpline == '-1' or dpline == 'EOF':
                    break
                if dpline:
                    depot_candidate = int(dpline)
                    if depot is None:
                        depot = depot_candidate
                    else:
                        raise ValueError(f"Multiple depots specified: {depot}, {depot_candidate}")
                i += 1
            continue
        i += 1

    for sec, found in required_sections.items():
        if not found:
            raise ValueError(f"Missing required section: {sec.replace('_', ' ').title()}")
    if dimension is None:
        raise ValueError("Missing DIMENSION field.")
    if vehicle_capacity is None:
        raise ValueError("Missing CAPACITY field.")
    if depot is None:
        raise ValueError("Depot must be specified.")
    if not (len(node_coordinates) == len(demands) == len(service_times) == len(time_windows) == dimension):
        raise ValueError("Input data lists do not match DIMENSION.")

    result = {
        'depot': depot,
        'node_coordinates': [coords[:] for coords in node_coordinates],
        'demands': list(demands),
        'vehicle_capacity': [vehicle_capacity],
        'service_times': list(service_times),
        'time_windows': [tw[:] for tw in time_windows]
    }
    return result

def distance(coords):
    n = len(coords)
    dist_matrix = []
    for i in range(n):
        row = []
        x1, y1 = coords[i]
        for j in range(n):
            x2, y2 = coords[j]
            row.append(math.hypot(x1 - x2, y1 - y2))
        dist_matrix.append(row)
    return dist_matrix

def cost(route, dist_matrix):
    total_cost = 0
    def route_cost(r):
        cc = 0
        for i in range(len(r) - 1):
            from_idx = r[i] - 1
            to_idx = r[i+1] - 1
            cc += dist_matrix[from_idx][to_idx]
        return cc
    if route and isinstance(route[0], list):
        for r in route:
            total_cost += route_cost(r)
        return total_cost
    elif isinstance(route, list):
        return route_cost(route)
    return 0

def initial(instance, dist_matrix):
    depot_id = instance["depot"]
    n_nodes = len(instance["node_coordinates"])
    demands = instance["demands"]
    capacity = instance["vehicle_capacity"][0]
    service_times = instance["service_times"]
    time_windows = instance["time_windows"]

    customers = set(range(1, n_nodes + 1))
    customers.remove(depot_id)
    unrouted = set(customers)
    routes = []

    while unrouted:
        route = [depot_id]
        curr_node = depot_id
        curr_capacity = 0.0
        curr_time = max(time_windows[depot_id - 1][0], 0.0)
        feasible_added = True
        curr_capacity = 0.0 # fix: reset capacity before starting new route
        while feasible_added and unrouted:
            feasible_added = False
            best_cand = None
            min_cost = float('inf')
            best_times = None
            for cand in unrouted:
                demand = demands[cand - 1]
                if curr_capacity + demand - 1e-7 > capacity:
                    continue
                travel_time = dist_matrix[curr_node - 1][cand - 1]
                arrival_time = curr_time + travel_time
                window_start, window_end = time_windows[cand - 1]
                if arrival_time - 1e-7 > window_end:
                    continue
                begin_service = max(arrival_time, window_start)
                depart_time = begin_service + service_times[cand - 1]
                return_time = dist_matrix[cand - 1][depot_id - 1]
                depot_window_start, depot_window_end = time_windows[depot_id - 1]
                if depart_time + return_time - 1e-7 > depot_window_end:
                    continue
                if arrival_time < window_start - 1e-7:
                    begin_service = window_start
                else:
                    begin_service = arrival_time
                if depart_time < -1e-8:
                    continue
                route_incr = dist_matrix[curr_node - 1][cand - 1]
                if route_incr < min_cost:
                    min_cost = route_incr
                    best_cand = cand
                    best_times = (begin_service, depart_time, demand)
            if best_cand is not None:
                route.append(best_cand)
                curr_node = best_cand
                curr_time = best_times[1]
                curr_capacity += best_times[2]
                unrouted.remove(best_cand)
                feasible_added = True
        route.append(depot_id)
        routes.append(route)
    return routes

def destroy(instance, dist_matrix, solution, ratio):
    import copy
    depot_id = instance['depot']
    n_nodes = len(instance['node_coordinates'])
    all_customers = [i for i in range(1, n_nodes + 1) if i != depot_id]
    to_remove_count = int(len(all_customers) * ratio)
    to_remove_count = max(1, to_remove_count)
    routes = copy.deepcopy(solution)
    removed_nodes = []
    removed_nodes_set = set()
    destroyed_route_indices = set()
    route_pos_list = []
    for r_idx, route in enumerate(routes):
        for idx in range(1, len(route) - 1):
            cust_id = route[idx]
            if cust_id != depot_id:
                route_pos_list.append((r_idx, idx, cust_id))
    customers_not_depot = [cid for _, _, cid in route_pos_list]
    if not customers_not_depot:
        return [], routes
    center_customer = random.choice(customers_not_depot)
    center_coord = instance['node_coordinates'][center_customer - 1]
    dist_to_center = []
    for ridx, pos, cust in route_pos_list:
        if cust not in removed_nodes_set:
            coord = instance['node_coordinates'][cust - 1]
            d = math.hypot(coord[0] - center_coord[0], coord[1] - center_coord[1])
            dist_to_center.append((d, ridx, pos, cust))
    dist_to_center.sort()
    cur_num_removed = 0
    for _, ridx, pos, cand in dist_to_center:
        if ridx in destroyed_route_indices:
            continue
        if cur_num_removed >= to_remove_count:
            break
        route = routes[ridx]
        seg_start, seg_end = pos, pos
        while seg_start > 1 and route[seg_start - 1] != depot_id and route[seg_start - 1] not in removed_nodes_set:
            seg_start -= 1
        while seg_end < len(route) - 2 and route[seg_end + 1] != depot_id and route[seg_end + 1] not in removed_nodes_set:
            seg_end += 1
        max_len = min(seg_end - seg_start + 1, to_remove_count - cur_num_removed)
        possible_lens = [l for l in range(1, max_len + 1)]
        if not possible_lens:
            continue
        subseq_len = random.choice(possible_lens)
        lb = max(seg_start, pos - (subseq_len - 1))
        ub = min(seg_end - subseq_len + 1, pos)
        possible_starts = [i for i in range(lb, ub + 1)]
        if not possible_starts:
            possible_starts = [seg_start]
        start_idx = random.choice(possible_starts)
        end_idx = start_idx + subseq_len - 1
        subseq = route[start_idx:(end_idx + 1)]
        if depot_id in subseq:
            continue
        if any(n in removed_nodes_set for n in subseq):
            continue
        for n in subseq:
            removed_nodes.append(n)
            removed_nodes_set.add(n)
        routes[ridx] = route[:start_idx] + route[end_idx + 1:]
        destroyed_route_indices.add(ridx)
        cur_num_removed += len(subseq)
    destroyed_solution = []
    for route in routes:
        cleaned = []
        prev_is_depot = False
        for node in route:
            if node == depot_id:
                if not prev_is_depot:
                    cleaned.append(node)
                prev_is_depot = True
            else:
                cleaned.append(node)
                prev_is_depot = False
        if len(cleaned) == 0:
            continue
        if cleaned[0] != depot_id:
            cleaned = [depot_id] + cleaned
        if cleaned[-1] != depot_id:
            cleaned = cleaned + [depot_id]
        if any(n != depot_id for n in cleaned):
            destroyed_solution.append(cleaned)
    destroyed_solution = [route for route in destroyed_solution if any(nid != depot_id for nid in route)]
    destroyed_nodes_set = set()
    for route in destroyed_solution:
        destroyed_nodes_set.update(n for n in route if n != depot_id)
    total_customer_set = destroyed_nodes_set.union(removed_nodes_set)
    if destroyed_nodes_set.intersection(removed_nodes_set):
        raise RuntimeError("Customer assigned to both destroyed_solution and removed_nodes")
    if set(all_customers) != total_customer_set:
        raise RuntimeError("Destroyed/removed nodes do not partition all customers")
    if depot_id in removed_nodes or depot_id in removed_nodes_set:
        raise RuntimeError("Depot present in removed nodes")
    return removed_nodes, destroyed_solution

def insert(destroyed_solution, removed_nodes, instance, dist_matrix):
    import copy
    depot_id = instance["depot"]
    demands = instance["demands"]
    capacity = instance["vehicle_capacity"][0]
    service_times = instance["service_times"]
    time_windows = instance["time_windows"]
    solution = copy.deepcopy(destroyed_solution)
    inserted = set()
    n_nodes = len(instance["node_coordinates"])
    def check_feasibility(route):
        cap = 0.0
        t = max(time_windows[depot_id - 1][0], 0.0)
        depot_tw_start, depot_tw_end = time_windows[depot_id - 1]
        for idx in range(1, len(route)):
            prev = route[idx - 1]
            node = route[idx]
            if node == depot_id:
                if idx == 1:
                    cap = 0.0
                    t = max(time_windows[depot_id - 1][0], 0.0)
                else:
                    if t - 1e-7 > depot_tw_end:
                        return False
                    cap = 0.0
                    t = max(time_windows[depot_id - 1][0], 0.0)
            else:
                cap += demands[node - 1]
                if cap - 1e-7 > capacity:
                    return False
                tt = dist_matrix[prev - 1][node - 1]
                arrival = t + tt
                win_start, win_end = time_windows[node - 1]
                if arrival - 1e-7 > win_end:
                    return False
                begin_service = max(arrival, win_start)
                t = begin_service + service_times[node - 1]
        if t - 1e-7 > time_windows[depot_id - 1][1]:
            return False
        return True
    for node in removed_nodes:
        if node in inserted:
            continue
        min_incr = None
        best_route = None
        best_pos = None
        for r_idx, route in enumerate(solution):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [node] + route[pos:]
                in_route = [n for n in new_route if n != depot_id]
                if len(set(in_route)) < len(in_route):
                    continue
                cap_sum = 0.0
                feasible = True
                for k in range(1, len(new_route) -1):
                    n = new_route[k]
                    if n == depot_id:
                        cap_sum = 0.0
                    else:
                        cap_sum += demands[n -1]
                        if cap_sum - 1e-7 > capacity:
                            feasible = False
                            break
                if not feasible:
                    continue
                if check_feasibility(new_route):
                    d_old = dist_matrix[route[pos - 1] - 1][route[pos] - 1]
                    d_new = dist_matrix[route[pos - 1] - 1][node - 1] + dist_matrix[node - 1][route[pos] - 1]
                    incr = d_new - d_old
                    if min_incr is None or incr < min_incr:
                        min_incr = incr
                        best_route = r_idx
                        best_pos = pos
        if best_route is not None:
            route = solution[best_route]
            solution[best_route] = route[:best_pos] + [node] + route[best_pos:]
            inserted.add(node)
            continue
        new_r = [depot_id, node, depot_id]
        if check_feasibility(new_r):
            solution.append(new_r)
            inserted.add(node)
        else:
            raise RuntimeError(f"Cannot insert node {node!r} feasibly")
    final_routes = []
    for r in solution:
        nods = [x for x in r if x != depot_id]
        if any(nid != depot_id for nid in r):
            out = r
            if out[0] != depot_id:
                out = [depot_id] + out
            if out[-1] != depot_id:
                out = out + [depot_id]
            final_routes.append(out)
    return final_routes

def validate(solution, instance, dist_matrix):
    depot_id = instance["depot"]
    n_nodes = len(instance["node_coordinates"])
    demands = instance["demands"]
    capacity = instance["vehicle_capacity"][0]
    service_times = instance["service_times"]
    time_windows = instance["time_windows"]

    for r_idx, route in enumerate(solution):
        if len(route) < 2 or route[0] != depot_id or route[-1] != depot_id:
            print(f"Depot constraint violated in route {r_idx}: does not start and end at depot.")
            return False

    customer_set = set(range(1, n_nodes + 1))
    customer_set.discard(depot_id)
    visited = []
    for route in solution:
        for node in route:
            if node != depot_id:
                visited.append(node)
            elif node == depot_id and route.count(node) > 2:
                print("Depot constraint violated: depot appears more than twice in a route.")
                return False
    v_set = set(visited)
    if v_set != customer_set:
        missing = customer_set - v_set
        extra = v_set - customer_set
        if missing:
            print(f"Not all customers are visited exactly once. Missing: {sorted(missing)}")
        if extra:
            print(f"Invalid node IDs found in routes: {sorted(extra)}")
        return False
    if len(visited) != len(customer_set):
        print(f"Some customers are visited more than once.")
        return False

    for r_idx, route in enumerate(solution):
        cap = 0.0
        t = max(time_windows[depot_id - 1][0], 0.0)
        depot_tw_start, depot_tw_end = time_windows[depot_id - 1]
        for idx in range(1, len(route)):
            prev = route[idx - 1]
            node = route[idx]
            if node == depot_id:
                if idx != 1 and t - 1e-7 > depot_tw_end:
                    print(f"Time window violated for depot at end of route {r_idx}. Departure/return time: {t}, window: {depot_tw_start}-{depot_tw_end}")
                    return False
                cap = 0.0
                t = max(depot_tw_start, 0.0)
            else:
                cap += demands[node - 1]
                if cap - 1e-7 > capacity:
                    print(f"Capacity constraint violated in route {r_idx} at node {node}. Load: {cap} > Capacity: {capacity}")
                    return False
                tt = dist_matrix[prev - 1][node - 1]
                arrival = t + tt
                win_start, win_end = time_windows[node - 1]
                if arrival - 1e-7 > win_end:
                    print(f"Time window violated at node {node} in route {r_idx}. Arrival: {arrival}, Window: {win_start}-{win_end}")
                    return False
                begin_service = max(arrival, win_start)
                t = begin_service + service_times[node - 1]
        if t - 1e-7 > depot_tw_end:
            print(f"Time window violated for depot on route {r_idx} end. Arrival: {t}, Window: {depot_tw_start}-{depot_tw_end}")
            return False

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to .vrp file')
    parser.add_argument('--iteration', type=int, default=100, help='Number of iterations (default: 100)')
    args = parser.parse_args()

    path = args.path
    iteration = args.iteration

    best_solution = None
    best_cost = float('inf')

    instance = read_vrp(path)
    coords = instance['node_coordinates']
    dist_matrix = distance(coords)
    current_solution = initial(instance, dist_matrix)
    if not validate(solution=current_solution, instance=instance, dist_matrix=dist_matrix):
        raise Exception("Initial solution is not valid!")
    current_cost = cost(route=current_solution, dist_matrix=dist_matrix)
    best_solution = current_solution
    best_cost = current_cost

    for step in range(iteration):
        ratio = random.uniform(0.05, 0.2) # fix: avoid degenerate ratio for destroy phase
        removed_nodes, destroyed_solution = destroy(instance, dist_matrix, solution=current_solution, ratio=ratio)
        current_solution = insert(destroyed_solution, removed_nodes, instance, dist_matrix)
        if not validate(solution=current_solution, instance=instance, dist_matrix=dist_matrix):
            raise Exception(f"Invalid solution encountered at iteration {step}!")
        current_cost = cost(route=current_solution, dist_matrix=dist_matrix)
        if current_cost <= best_cost:
            best_solution = current_solution
            best_cost = current_cost
        else:
            p = random.uniform(0, 1)
            threshold = math.exp(-(current_cost - best_cost) * iteration * 10 / (iteration - step + 1))
            if p > threshold:
                current_solution = best_solution
                current_cost = best_cost

    print(f"the process is successful, the best cost is {best_cost}")