import re
import math
import random
import copy
import argparse
import sys

def read_vrp(path: str):
    depot = []
    node_coordinates = []
    demands = []
    service_times = []
    time_windows = []
    vehicle_capacity = []

    with open(path, 'r') as file:
        lines = file.readlines()

    dimension = None
    capacity = None

    section_headers = [
        'NODE_COORD_SECTION',
        'DEMAND_SECTION',
        'TIME_WINDOW_SECTION',
        'SERVICE_TIME_SECTION',
        'DEPOT_SECTION'
    ]
    sections = {h: [] for h in section_headers}
    current_section = None

    for line in lines:
        ln = line.strip()
        if ln.startswith('DIMENSION'):
            m = re.search(r'(\d+)', ln)
            if m:
                dimension = int(m.group(1))
        elif ln.startswith('CAPACITY'):
            m = re.search(r'(\d+)', ln)
            if m:
                capacity = int(m.group(1))

    for line in lines:
        ln = line.strip()
        if ln in section_headers:
            current_section = ln
            continue
        if ln == '-1' or ln == 'EOF' or ln == '':
            continue
        if current_section in section_headers:
            sections[current_section].append(ln)

    if sections['DEPOT_SECTION']:
        depot_id = int(sections['DEPOT_SECTION'][0])
        depot = [depot_id]

    def sort_by_id(section):
        return sorted(
            [l.split() for l in section],
            key=lambda x: int(x[0])
        )

    coords_info = sort_by_id(sections['NODE_COORD_SECTION'])
    node_coordinates = [[float(x[1]), float(x[2])] for x in coords_info]

    dem_info = sort_by_id(sections['DEMAND_SECTION'])
    demands = [float(x[1]) for x in dem_info]

    serv_info = sort_by_id(sections['SERVICE_TIME_SECTION'])
    service_times = [float(x[1]) for x in serv_info]

    tw_info = sort_by_id(sections['TIME_WINDOW_SECTION'])
    time_windows = [[float(x[1]), float(x[2])] for x in tw_info]

    if capacity is not None:
        vehicle_capacity = [capacity]

    assert len(node_coordinates) == dimension
    assert len(demands) == dimension
    assert len(service_times) == dimension
    assert len(time_windows) == dimension
    assert depot and depot[0] >= 1

    return {
        'depot': depot,
        'node_coordinates': node_coordinates,
        'demands': demands,
        'vehicle_capacity': vehicle_capacity,
        'service_times': service_times,
        'time_windows': time_windows
    }

def distance(coords):
    n = len(coords)
    dist_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.hypot(dx, dy)
            row.append(d)
        dist_matrix.append(row)
    return dist_matrix

def cost(route, dist_matrix):
    total_cost = 0.0
    if isinstance(route[0], list):
        # route is list of routes
        for single_route in route:
            for i in range(len(single_route) - 1):
                total_cost += dist_matrix[single_route[i] - 1][single_route[i + 1] - 1]
    else:
        for i in range(len(route) - 1):
            total_cost += dist_matrix[route[i] - 1][route[i + 1] - 1]
    return total_cost

def initial(instance, dist_matrix):
    num_nodes = len(instance['node_coordinates'])
    depot_id = instance['depot'][0]
    depot_idx = depot_id - 1
    capacity = float(instance['vehicle_capacity'][0])
    demands = instance['demands']
    service_times = instance['service_times']
    time_windows = instance['time_windows']

    unvisited = set(range(1, num_nodes + 1))
    unvisited.remove(depot_id)
    routes = []
    node_earliest = [tw[0] for tw in time_windows]
    node_latest = [tw[1] for tw in time_windows]

    while unvisited:
        route = [depot_id]
        curr_load = 0.0
        curr_time = node_earliest[depot_idx]
        curr_idx = depot_idx

        while True:
            feasibles = []
            for cust in unvisited:
                cidx = cust - 1
                if curr_load + demands[cidx] > capacity:
                    continue
                arrival_time = curr_time + dist_matrix[curr_idx][cidx]
                start_service = max(arrival_time, node_earliest[cidx])
                end_service = start_service + service_times[cidx]
                if start_service > node_latest[cidx]:
                    continue
                feasibles.append((cust, cidx, dist_matrix[curr_idx][cidx], start_service, end_service, arrival_time))

            if feasibles:
                best_from_current = min(feasibles, key=lambda x: x[2])
                depot_feasibles = []
                for cust in unvisited:
                    cidx = cust - 1
                    if demands[cidx] > capacity:
                        continue
                    depot_arrival = node_earliest[depot_idx] + dist_matrix[depot_idx][cidx]
                    depot_start_service = max(depot_arrival, node_earliest[cidx])
                    depot_end_service = depot_start_service + service_times[cidx]
                    if depot_start_service > node_latest[cidx]:
                        continue
                    depot_feasibles.append((cust, cidx, dist_matrix[depot_idx][cidx], depot_start_service, depot_end_service, depot_arrival))
                if depot_feasibles:
                    min_depot_dist = min(depot_feasibles, key=lambda x: x[2])[2]
                    min_current_dist = best_from_current[2]
                    if min_depot_dist < min_current_dist:
                        break
                cust, cidx, _, svc_start, svc_end, arr_time = best_from_current
                route.append(cust)
                curr_load += demands[cidx]
                curr_time = svc_end
                curr_idx = cidx
                unvisited.remove(cust)
            else:
                break

        if len(route) > 1:
            routes.append(route)

    for route in routes:
        assert route[0] == depot_id
        assert depot_id not in route[1:]
    return routes

def destroy(instance, dist_matrix, solution, ratio):
    n = len(instance['node_coordinates'])
    depot_id = instance['depot'][0]
    all_customers = [i for i in range(1, n+1) if i != depot_id]
    nb_remove = int(len(all_customers) * ratio)
    nb_remove = max(1, nb_remove)
    customers_set = set(all_customers)

    destroyed_solution = copy.deepcopy(solution)
    removed_nodes = []

    destroyed_route_indices = set()
    removed_nodes_set = set()

    candidate_centers = list(customers_set)
    center_node = random.choice(candidate_centers)
    center_idx = center_node - 1

    neighbor_nodes = []
    for node in all_customers:
        d = dist_matrix[center_idx][node-1] if node != center_node else 0.0
        neighbor_nodes.append((d, node))
    neighbor_nodes.sort()
    neighbor_order = [node for _,node in neighbor_nodes]

    route_lookup = {}
    for r_idx, route in enumerate(destroyed_solution):
        for pos, node in enumerate(route):
            if node != depot_id:
                route_lookup[node] = (r_idx, pos)

    used_routes = set()
    neighbor_cursor = 0
    while len(removed_nodes) < nb_remove and neighbor_cursor < len(neighbor_order):
        node = neighbor_order[neighbor_cursor]
        if (node == depot_id) or (node in removed_nodes_set):
            neighbor_cursor += 1
            continue
        node_info = route_lookup.get(node, None)
        if node_info is None:
            neighbor_cursor += 1
            continue
        r_idx, pos_in_route = node_info
        if r_idx in used_routes:
            neighbor_cursor += 1
            continue
        route = destroyed_solution[r_idx]

        customer_indices = [i for i in range(len(route)) if route[i] != depot_id]
        cust_pos = None
        customer_nodes = []
        for ind in customer_indices:
            customer_nodes.append(route[ind])
        for c, ind in zip(customer_nodes, customer_indices):
            if c == node:
                cust_pos = ind
                break
        if cust_pos is None:
            neighbor_cursor += 1
            continue

        max_length = len(customer_indices)
        possible_subseqs = []
        min_idx = customer_indices[0]
        max_idx = customer_indices[-1]
        for L in range(1, max_length+1):
            for start in range(min_idx, cust_pos+1):
                end = start + L - 1
                if end > max_idx:
                    continue
                if (cust_pos < start) or (cust_pos > end):
                    continue
                subseq = route[start:end+1]
                if depot_id in subseq:
                    continue
                possible_subseqs.append( (start, end) )
        if not possible_subseqs:
            neighbor_cursor += 1
            continue
        subseq_start, subseq_end = random.choice(possible_subseqs)
        subseq_nodes = route[subseq_start:subseq_end+1]

        for n in subseq_nodes:
            if n == depot_id:
                continue
            if n not in removed_nodes_set:
                removed_nodes.append(n)
                removed_nodes_set.add(n)
                customers_set.discard(n)
        new_route = route[:subseq_start] + route[subseq_end+1:]
        destroyed_solution[r_idx] = new_route

        used_routes.add(r_idx)
        neighbor_cursor += 1

    destroyed_solution = [
        [node for node in r if node != depot_id or (node == depot_id and r.index(node)==0)]
        for r in destroyed_solution if any(x != depot_id for x in r)
    ]
    merged_destroyed_solution = []
    for route in destroyed_solution:
        if not route:
            continue
        merged_route = []
        prev = None
        for node in route:
            if node == depot_id:
                if prev != depot_id:
                    merged_route.append(node)
            else:
                merged_route.append(node)
            prev = node
        merged_destroyed_solution.append(merged_route)
    destroyed_solution = merged_destroyed_solution

    remaining_nodes = set()
    for route in destroyed_solution:
        for node in route:
            if node != depot_id:
                remaining_nodes.add(node)
    removed_set = set(removed_nodes)
    assert removed_set.isdisjoint(set([depot_id]))
    assert set(all_customers) == remaining_nodes.union(removed_set)
    assert not (remaining_nodes & removed_set)
    assert depot_id not in removed_nodes

    return removed_nodes, destroyed_solution

def insert(destroyed_solution, removed_nodes, instance, dist_matrix):
    depot_id = instance['depot'][0]
    capacity = float(instance['vehicle_capacity'][0])
    demands = instance['demands']
    service_times = instance['service_times']
    time_windows = instance['time_windows']
    n = len(demands)
    node_earliest = [tw[0] for tw in time_windows]
    node_latest = [tw[1] for tw in time_windows]
    solution = [r[:] for r in destroyed_solution]
    inserted = set()
    for node in removed_nodes:
        best_delta = None
        best_route_idx = None
        best_pos = None
        best_new_route = None
        best_is_new_route = False

        node_idx = node - 1
        demand = demands[node_idx]
        for r_idx, route in enumerate(solution):
            possible_positions = list(range(1, len(route)+1))
            for pos in possible_positions:
                temp_route = route[:pos] + [node] + route[pos:]
                if any(
                        (i != 0 and temp_route[i] == depot_id)
                        for i in range(len(temp_route))
                ):
                    continue
                load = sum(demands[n-1] for n in temp_route if n != depot_id)
                if load > capacity:
                    continue
                feasible = True
                arr_times = [0.0] * len(temp_route)
                beg_window = node_earliest
                end_window = node_latest
                arr_times[0] = beg_window[temp_route[0]-1]
                t = arr_times[0]
                for i in range(1, len(temp_route)):
                    prev = temp_route[i-1]-1
                    cur = temp_route[i]-1
                    t = max(arr_times[i-1], beg_window[prev]) + (service_times[prev] if i>1 else 0.0)
                    t += dist_matrix[prev][cur]
                    start_service = max(t, beg_window[cur])
                    finish_service = start_service + service_times[cur]
                    arr_times[i] = start_service
                    if start_service > end_window[cur]:
                        feasible = False
                        break
                if not feasible:
                    continue
                oldcost = 0.0
                for i in range(len(route)-1):
                    oldcost += dist_matrix[route[i]-1][route[i+1]-1]
                newcost = 0.0
                for i in range(len(temp_route)-1):
                    newcost += dist_matrix[temp_route[i]-1][temp_route[i+1]-1]
                delta = newcost - oldcost
                if (best_delta is None) or (delta < best_delta):
                    best_delta = delta
                    best_route_idx = r_idx
                    best_pos = pos
                    best_new_route = temp_route
                    best_is_new_route = False
        new_route = [depot_id, node]
        load = demand
        if load <= capacity:
            arr_times = [node_earliest[depot_id-1], 0.0]
            feasible = True
            prev = depot_id-1
            cur = node_idx
            t = node_earliest[prev]
            t += dist_matrix[prev][cur]
            start_service = max(t, node_earliest[cur])
            finish_service = start_service + service_times[cur]
            arr_times[1] = start_service
            if start_service <= node_latest[cur]:
                new_cost = dist_matrix[depot_id-1][node_idx]
                delta = new_cost
                if (best_delta is None) or (delta < best_delta):
                    best_delta = delta
                    best_route_idx = len(solution)
                    best_pos = 1
                    best_new_route = new_route
                    best_is_new_route = True
        if best_new_route is not None and not best_is_new_route:
            cleaned = [x for idx, x in enumerate(best_new_route) if (x != depot_id or idx == 0)]
            solution[best_route_idx] = cleaned
        elif best_new_route is not None and best_is_new_route:
            solution.append([depot_id, node])
        else:
            solution.append([depot_id, node])
        inserted.add(node)
    solution = [r for r in solution if any(n != depot_id for n in r)]
    merged = []
    for route in solution:
        if not route:
            continue
        merged_route = []
        last = None
        for n in route:
            if n == depot_id:
                if last != depot_id:
                    merged_route.append(n)
            else:
                merged_route.append(n)
            last = n
        if merged_route and merged_route[0] == depot_id and (len(merged_route) == 1 or merged_route[-1] != depot_id):
            merged.append(merged_route)
        elif merged_route and merged_route[0] == depot_id:
            cleaned = [merged_route[0]] + [x for x in merged_route[1:] if x != depot_id]
            merged.append(cleaned)
    return merged

def validate(solution, instance, dist_matrix):
    depot_id = instance['depot'][0]
    n = len(instance['node_coordinates'])
    capacity = float(instance['vehicle_capacity'][0])
    demands = instance['demands']
    service_times = instance['service_times']
    time_windows = instance['time_windows']
    customers = set(range(1, n+1))
    customers.discard(depot_id)

    seen = set()
    for route in solution:
        for node in route:
            if node == depot_id:
                continue
            if node in seen:
                print(f"Visit constraint violated: Node {node} visited more than once.")
                return False
            seen.add(node)
    if seen != customers:
        missed = customers - seen
        print(f"Visit constraint violated: Not all customers visited. Missed: {missed}")
        return False

    for rix, route in enumerate(solution):
        load = 0.0
        if len(route) == 0 or route[0] != depot_id:
            print(f"Open route constraint violated: Route {rix+1} does not start at depot.")
            return False
        depot_count = route.count(depot_id)
        if depot_count > 1:
            print(f"Open route constraint violated: Depot occurs {depot_count} times in route {rix+1}.")
            return False
        if len(route) > 1 and route[-1] == depot_id:
            print(f"Open route constraint violated: Route {rix+1} returns to depot at end.")
            return False
        time = time_windows[depot_id-1][0]
        prev = depot_id - 1
        for pos in range(1, len(route)):
            cur = route[pos] - 1
            dmd = demands[cur]
            load += dmd
            time += dist_matrix[prev][cur]
            earliest, latest = time_windows[cur]
            time = max(time, earliest)
            if time > latest + 1e-6:
                print(f"Time window constraint violated at node {route[pos]} in route {rix+1}: arrival {time:.3f} > latest {latest:.3f}")
                return False
            time += service_times[cur]
            prev = cur
        if load - capacity > 1e-6:
            print(f"Capacity constraint violated in route {rix+1}: load {load:.3f} > capacity {capacity:.3f}")
            return False

    for rix, route in enumerate(solution):
        if len(route) > 0:
            if route[0] != depot_id:
                print(f"Open route constraint violated: Route {rix+1} does not start at depot.")
                return False
            depot_count = route.count(depot_id)
            if depot_count > 1:
                print(f"Open route constraint violated: Depot occurs more than once in route {rix+1}.")
                return False
            if len(route) > 1 and route[-1] == depot_id:
                print(f"Open route constraint violated: Route {rix+1} returns to depot at end.")
                return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to .vrp file')
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

    valid = validate(solution=current_solution, instance=instance, dist_matrix=dist_matrix)
    if not valid:
        raise Exception("Initial solution is invalid")
    current_cost = cost(route=current_solution, dist_matrix=dist_matrix)
    best_solution = current_solution
    best_cost = current_cost

    # Simulated annealing parameters
    initial_temperature = 1.0
    final_temperature = 1e-4
    alpha = 0.995  # cooling rate
    temperature = initial_temperature

    for step in range(iteration):
        ratio = random.uniform(0.00001, 0.2)
        removed_nodes, destroyed_solution = destroy(instance, dist_matrix, solution=current_solution, ratio=ratio)
        current_solution = insert(destroyed_solution, removed_nodes, instance, dist_matrix)
        valid = validate(solution=current_solution, instance=instance, dist_matrix=dist_matrix)
        if not valid:
            raise Exception("Solution after destroy-insert is invalid")
        current_cost = cost(route=current_solution, dist_matrix=dist_matrix)
        if current_cost <= best_cost:
            best_solution = current_solution
            best_cost = current_cost
        else:
            delta = current_cost - best_cost
            try:
                if temperature > final_temperature:
                    threshold = math.exp(-delta / temperature)
                else:
                    threshold = 0.0
            except OverflowError:
                threshold = 0.0
            p = random.random()
            if p > threshold:
                current_solution = best_solution
                current_cost = best_cost
        temperature = max(final_temperature, temperature * alpha)

    print(f"the process is successful, the best cost is {best_cost}")