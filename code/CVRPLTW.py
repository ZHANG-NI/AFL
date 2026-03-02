import re
import math
import random
import argparse
import sys

def read_vrp(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()

    node_coords = {}
    demands = {}
    time_windows = {}
    service_times = {}
    depot = []
    vehicle_capacity = None
    distance_limits = None

    section = None
    for line in lines:
        l = line.strip()
        if not l:
            continue
        if l.startswith('DIMENSION'):
            continue
        elif l.startswith('CAPACITY'):
            value = float(l.split(':')[-1].strip())
            vehicle_capacity = value
            continue
        elif l.startswith('DISTANCE_LIMIT'):
            value = float(l.split(':')[-1].strip())
            distance_limits = value
            continue
        elif l.startswith('NODE_COORD_SECTION'):
            section = 'node_coord'
            continue
        elif l.startswith('DEMAND_SECTION'):
            section = 'demand'
            continue
        elif l.startswith('TIME_WINDOW_SECTION'):
            section = 'tw'
            continue
        elif l.startswith('SERVICE_TIME_SECTION'):
            section = 'service'
            continue
        elif l.startswith('DEPOT_SECTION'):
            section = 'depot'
            continue
        elif l.startswith('-1'):
            section = None
            continue
        elif l.startswith('EOF'):
            break
        else:
            if section == 'node_coord':
                vals = l.split()
                idx = int(vals[0])
                node_coords[idx] = [float(vals[1]), float(vals[2])]
            elif section == 'demand':
                vals = l.split()
                idx = int(vals[0])
                demands[idx] = float(vals[1])
            elif section == 'tw':
                vals = l.split()
                idx = int(vals[0])
                time_windows[idx] = [float(vals[1]), float(vals[2])]
            elif section == 'service':
                vals = l.split()
                idx = int(vals[0])
                service_times[idx] = float(vals[1])
            elif section == 'depot':
                idx = int(l)
                if idx != -1:
                    depot.append(idx)

    all_node_ids = sorted(node_coords.keys())
    N = len(all_node_ids)
    assert depot, "No depot found in instance."
    depot_id = depot[0]
    assert depot_id in all_node_ids

    # Defensive: require all node ids in all definitions
    for nid in all_node_ids:
        assert nid in node_coords, f"Missing coords for node {nid}"
        assert nid in demands, f"Missing demand for node {nid}"
        assert nid in time_windows, f"Missing time window for node {nid}"
        assert nid in service_times, f"Missing service time for node {nid}"

    node_coordinates = [node_coords[nid] for nid in all_node_ids]
    demands_list = [demands[nid] for nid in all_node_ids]
    service_times_list = [service_times[nid] for nid in all_node_ids]
    time_windows_list = [time_windows[nid] for nid in all_node_ids]

    output = {
        'depot': [depot_id],
        'node_coordinates': node_coordinates,
        'demands': demands_list,
        'vehicle_capacity': vehicle_capacity,
        'time_windows': time_windows_list,
        'service_times': service_times_list,
        'distance_limits': distance_limits,
        'node_id_list': all_node_ids,
        'node_id_to_idx': {nid: i for i, nid in enumerate(all_node_ids)},
        'idx_to_node_id': {i: nid for i, nid in enumerate(all_node_ids)},
    }
    # Restrict to output keys as required by the specification
    return {k: v for k, v in output.items() if k in [
        'depot','node_coordinates','demands','vehicle_capacity','time_windows','service_times','distance_limits','node_id_list','node_id_to_idx','idx_to_node_id'
    ]}

def distance(coords):
    n = len(coords)
    dist_matrix = []
    for i in range(n):
        row = []
        x1, y1 = coords[i]
        for j in range(n):
            x2, y2 = coords[j]
            d = math.hypot(x1 - x2, y1 - y2)
            row.append(d)
        dist_matrix.append(row)
    return dist_matrix

def cost(route, dist_matrix, node_id_to_idx):
    total = 0.0
    for i in range(len(route)-1):
        u = route[i]
        v = route[i+1]
        total += dist_matrix[node_id_to_idx[u]][node_id_to_idx[v]]
    return total

def initial(instance, dist_matrix):
    node_id_list = instance['node_id_list']
    node_id_to_idx = instance['node_id_to_idx']
    depot_id = instance['depot'][0]
    customer_ids = [nid for nid in node_id_list if nid != depot_id]
    demands = {nid: instance['demands'][i] for i, nid in enumerate(node_id_list)}
    capacity = instance['vehicle_capacity']
    tw = {nid: instance['time_windows'][i] for i, nid in enumerate(node_id_list)}
    service_times = {nid: instance['service_times'][i] for i, nid in enumerate(node_id_list)}
    distance_limit = instance['distance_limits']

    unvisited = set(customer_ids)
    solution = []

    while unvisited:
        route = [depot_id]
        curr_id = depot_id
        curr_time = tw[curr_id][0]
        curr_load = 0.0
        curr_route_dist = 0.0
        prev_id = curr_id

        while True:
            candidates = []
            for cid in sorted(unvisited):
                cidx = node_id_to_idx[cid]
                dmd = demands[cid]
                travel_time = dist_matrix[node_id_to_idx[prev_id]][cidx]
                arr_time = curr_time + travel_time
                ready, due = tw[cid]
                stime = service_times[cid]
                leave_time = max(arr_time, ready) + stime
                new_load = curr_load + dmd
                trial_route = route + [cid, depot_id]
                trial_dist = 0.0
                for k in range(len(trial_route)-1):
                    trial_dist += dist_matrix[node_id_to_idx[trial_route[k]]][node_id_to_idx[trial_route[k+1]]]
                feasible = True
                if new_load > capacity + 1e-8:
                    feasible = False
                if leave_time > due + 1e-8:
                    feasible = False
                if trial_dist > distance_limit + 1e-8:
                    feasible = False
                if curr_time + travel_time > due + 1e-8:
                    feasible = False
                if not feasible:
                    continue
                candidates.append((dist_matrix[node_id_to_idx[prev_id]][cidx], cid, cidx, travel_time, arr_time, leave_time, trial_dist))
            if not candidates:
                break
            candidates.sort(key=lambda x: (x[0], x[1]))
            _, next_cid, next_idx, travel_time, arr_time, leave_time, route_dist = candidates[0]
            route.append(next_cid)
            unvisited.remove(next_cid)
            curr_load += demands[next_cid]
            curr_time = max(arr_time, tw[next_cid][0]) + service_times[next_cid]
            prev_id = next_cid
        route.append(depot_id)
        solution.append(route)

    return solution

def destroy(instance, dist_matrix, solution, ratio):
    depot_id = instance['depot'][0]
    node_id_list = instance['node_id_list']
    node_id_to_idx = instance['node_id_to_idx']
    all_customers = [nid for nid in node_id_list if nid != depot_id]
    n_destroy = int(len(all_customers)*ratio)
    if n_destroy < 1:
        n_destroy = 1
    n_destroy = min(n_destroy, len(all_customers))
    removed_nodes = []
    destroyed_solution = [route[:] for route in solution]
    destroyed_routes_indices = set()
    remaining_customers_set = set()
    for route in destroyed_solution:
        for nid in route:
            if nid != depot_id:
                remaining_customers_set.add(nid)
    if n_destroy > len(remaining_customers_set):
        n_destroy = len(remaining_customers_set)
    all_possible_to_remove = sorted(list(remaining_customers_set))
    random.shuffle(all_possible_to_remove)
    i_candidate = 0
    while len(removed_nodes) < n_destroy and i_candidate < len(all_possible_to_remove):
        nid = all_possible_to_remove[i_candidate]
        i_candidate += 1
        if nid in removed_nodes:
            continue
        found_route_idx = None
        found_pos = None
        for ridx, route in enumerate(destroyed_solution):
            try:
                pos = route.index(nid)
                found_route_idx = ridx
                found_pos = pos
                break
            except ValueError:
                continue
        if found_route_idx is None or found_pos is None:
            continue
        route = destroyed_solution[found_route_idx]
        destroyed_solution[found_route_idx] = [v for v in route if v != nid]
        removed_nodes.append(nid)

    final_solution = []
    for route in destroyed_solution:
        newroute = []
        prev_is_depot = False
        for nid in route:
            if nid == depot_id:
                if not prev_is_depot:
                    newroute.append(depot_id)
                    prev_is_depot = True
            else:
                newroute.append(nid)
                prev_is_depot = False
        if not newroute or newroute[0] != depot_id:
            newroute = [depot_id] + newroute
        if len(newroute) < 2 or newroute[-1] != depot_id:
            newroute.append(depot_id)
        cnt_cust = len([nid for nid in newroute if nid != depot_id])
        if cnt_cust >= 1:
            final_solution.append(newroute)
    destroyed_solution = final_solution

    new_customers_in_routes = set()
    for route in destroyed_solution:
        for nid in route:
            if nid != depot_id:
                new_customers_in_routes.add(nid)
    total_customers_now = set(removed_nodes).union(new_customers_in_routes)
    missing_customers = set(all_customers) - total_customers_now
    for m in missing_customers:
        removed_nodes.append(m)
    removed_nodes = [nid for nid in removed_nodes if nid in all_customers and (nid not in new_customers_in_routes)]
    assert set(removed_nodes).union(new_customers_in_routes) == set(all_customers)
    assert len(set(removed_nodes)) == len(removed_nodes)
    return removed_nodes, destroyed_solution

def insert(destroyed_solution, removed_nodes, instance, dist_matrix):
    solution = [route[:] for route in destroyed_solution]
    depot_id = instance['depot'][0]
    node_id_list = instance['node_id_list']
    node_id_to_idx = instance['node_id_to_idx']
    capacity = instance['vehicle_capacity']
    demands = {nid: instance['demands'][i] for i, nid in enumerate(node_id_list)}
    tw = {nid: instance['time_windows'][i] for i, nid in enumerate(node_id_list)}
    service_times = {nid: instance['service_times'][i] for i, nid in enumerate(node_id_list)}
    distance_limit = instance['distance_limits']

    def is_feasible_insertion(route):
        route_demands = sum([demands[nid] for nid in route if nid != depot_id])
        if route_demands > capacity + 1e-8:
            return False, None, None, None

        arr_times = [0.0] * len(route)
        leave_times = [0.0] * len(route)

        t = tw[depot_id][0]
        arr_times[0] = t
        leave_times[0] = t

        for i in range(1, len(route)):
            prev_id = route[i-1]
            curr_id = route[i]
            travel = dist_matrix[node_id_to_idx[prev_id]][node_id_to_idx[curr_id]]
            t = leave_times[i-1] + travel

            if curr_id == depot_id:
                arr_times[i] = t
                leave_times[i] = t
                continue
            ready, due = tw[curr_id]
            stime = service_times[curr_id]
            t_wait = max(t, ready)
            if t_wait > due + 1e-8:
                return False, None, None, None
            arr_times[i] = t
            leave_times[i] = t_wait + stime
            t = t_wait + stime

        total_dist = 0.0
        for i in range(len(route)-1):
            total_dist += dist_matrix[node_id_to_idx[route[i]]][node_id_to_idx[route[i+1]]]
        if total_dist > distance_limit + 1e-8:
            return False, None, None, None
        return True, route, arr_times, total_dist

    for node in removed_nodes:
        best_insertion = None
        min_incr = None
        best_route_idx = None
        best_new_route = None
        for ri, route in enumerate(solution):
            for pos in range(1, len(route)):
                prev_id = route[pos-1]
                next_id = route[pos]
                cost_before = dist_matrix[node_id_to_idx[prev_id]][node_id_to_idx[next_id]]
                cost_after = dist_matrix[node_id_to_idx[prev_id]][node_id_to_idx[node]] + dist_matrix[node_id_to_idx[node]][node_id_to_idx[next_id]]
                route_new = route[:pos] + [node] + route[pos:]
                feasible, _, _, _ = is_feasible_insertion(route_new)
                if feasible:
                    inc = cost_after - cost_before
                    if (min_incr is None) or (inc < min_incr):
                        min_incr = inc
                        best_insertion = pos
                        best_route_idx = ri
                        best_new_route = route_new
        if best_insertion is not None:
            solution[best_route_idx] = best_new_route
        else:
            route_new = [depot_id, node, depot_id]
            feasible, checked_route, arr_times, total_dist = is_feasible_insertion(route_new)
            if not feasible:
                raise RuntimeError(f"Node {node} cannot be feasibly reinserted")
            solution.append(route_new)
    return solution

def validate(solution, instance, dist_matrix):
    node_id_list = instance['node_id_list']
    node_id_to_idx = instance['node_id_to_idx']
    depot_ids = instance['depot']
    if not depot_ids:
        print("Depot not defined in instance.")
        return False
    depot_id = depot_ids[0]
    all_node_ids = set(node_id_list)
    customer_ids = set(nid for nid in node_id_list if nid != depot_id)
    demands = {nid: instance['demands'][i] for i, nid in enumerate(node_id_list)}
    capacity = instance['vehicle_capacity']
    distance_limit = instance['distance_limits']
    time_windows = {nid: instance['time_windows'][i] for i, nid in enumerate(node_id_list)}
    service_times = {nid: instance['service_times'][i] for i, nid in enumerate(node_id_list)}

    for rnum, route in enumerate(solution):
        if not route:
            print(f"Route {rnum} is empty.")
            return False
        if route[0] != depot_id:
            print(f"Route {rnum} does not start at depot.")
            return False
        if route[-1] != depot_id:
            print(f"Route {rnum} does not end at depot.")
            return False
        for i in range(1,len(route)-1):
            if route[i] == depot_id:
                print(f"Depot appears in the middle of route {rnum} at position {i}.")
                return False

    seen_customers = set()
    repeated = set()
    for rnum, route in enumerate(solution):
        for nid in route:
            if nid == depot_id:
                continue
            if nid not in customer_ids:
                print(f"Node {nid} in route {rnum} is not a customer.")
                return False
            if nid in seen_customers:
                repeated.add(nid)
            else:
                seen_customers.add(nid)
    missing = customer_ids.difference(seen_customers)
    if missing:
        print("Missing customers:", sorted(list(missing)))
        return False
    if repeated:
        print("Customers visited more than once:", sorted(list(repeated)))
        return False

    for rnum, route in enumerate(solution):
        load = 0.0
        for nid in route:
            if nid == depot_id:
                continue
            load += demands[nid]
        if load > capacity + 1e-8:
            print(f"Route {rnum} exceeds capacity: {load} > {capacity}")
            return False

    for rnum, route in enumerate(solution):
        total_dist = 0.0
        for i in range(len(route)-1):
            total_dist += dist_matrix[node_id_to_idx[route[i]]][node_id_to_idx[route[i+1]]]
        if total_dist > distance_limit + 1e-8:
            print(f"Route {rnum} exceeds the distance limit: {total_dist} > {distance_limit}")
            return False

    for rnum, route in enumerate(solution):
        depot_ready, depot_due = time_windows[depot_id]
        t = depot_ready
        arr_times = [t]
        leave_times = [t]
        for i in range(1, len(route)):
            prev = route[i-1]
            curr = route[i]
            travel = dist_matrix[node_id_to_idx[prev]][node_id_to_idx[curr]]
            t = leave_times[i-1] + travel
            if curr != depot_id:
                ready, due = time_windows[curr]
                st = service_times[curr]
                t_wait = max(t, ready)
                if t_wait > due + 1e-8:
                    print(f"Time window violated at node {curr} in route {rnum}; arrived {t:.5f}, window {ready}-{due}")
                    return False
                arr_times.append(t)
                leave_times.append(t_wait + st)
                t = t_wait + st
            else:
                arr_times.append(t)
                leave_times.append(t)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to .vrp file')
    parser.add_argument('--iteration', type=int, default=100, help='Number of iterations (default: 100)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()
    path = args.path
    iteration = args.iteration
    if args.seed is not None:
        random.seed(args.seed)

    best_solution = None
    best_cost = float('inf')

    instance = read_vrp(path)
    coords = instance['node_coordinates']
    dist_matrix = distance(coords)
    node_id_to_idx = instance['node_id_to_idx']
    current_solution = initial(instance, dist_matrix)
    valid = validate(current_solution, instance, dist_matrix)
    if not valid:
        raise Exception("Initial solution is invalid. Exiting.")
    current_cost = sum(cost(route, dist_matrix, node_id_to_idx) for route in current_solution)
    best_solution = current_solution
    best_cost = current_cost

    for step in range(iteration):
        ratio = random.uniform(0.0001, 0.2)
        removed_nodes, destroyed_solution = destroy(instance, dist_matrix, current_solution, ratio)
        current_solution = insert(destroyed_solution, removed_nodes, instance, dist_matrix)
        valid = validate(current_solution, instance, dist_matrix)
        if not valid:
            raise Exception("Solution after destroy/insert is invalid. Exiting.")
        current_cost = sum(cost(route, dist_matrix, node_id_to_idx) for route in current_solution)
        if current_cost <= best_cost:
            best_solution = [r[:] for r in current_solution]
            best_cost = current_cost
        else:
            p = random.uniform(0,1)
            delta = current_cost - best_cost
            temperature = max(1e-6, 1.0 * (1.0 - step / (iteration + 1)))
            try:
                threshold = math.exp(-delta / temperature) if delta > 0 else 1.0
            except OverflowError:
                threshold = 0.0
            if p < threshold:
                pass
            else:
                current_solution = [r[:] for r in best_solution]
                current_cost = best_cost

    print(f"the process is successful, the best cost is {best_cost}")