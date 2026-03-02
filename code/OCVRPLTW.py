import re
import math
import random
import argparse
import sys

def read_vrp(path: str):
    with open(path, 'r') as f:
        lines = [line.rstrip('\n') for line in f]

    re_keyval = re.compile(r'^(\w+)\s*:\s*(.*)')
    meta = {}
    section_map = {
        'NODE_COORD_SECTION': 'node_coordinates',
        'DEMAND_SECTION': 'demands',
        'TIME_WINDOW_SECTION': 'time_windows',
        'SERVICE_TIME_SECTION': 'service_times',
        'DEPOT_SECTION': 'depot'
    }
    sections = {
        "node_coordinates": [],
        "demands": [],
        "time_windows": [],
        "service_times": [],
        "depot": []
    }

    in_section = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        m = re_keyval.match(line)
        if m:
            key, val = m.group(1).upper(), m.group(2)
            meta[key] = val
            i += 1
            continue
        if line in section_map:
            in_section = section_map[line]
            i += 1
            continue
        if line == "EOF":
            break
        if in_section:
            if line == '-1':
                in_section = None
            else:
                sections[in_section].append(line)
            i += 1
            continue
        i += 1

    dimension = int(meta.get("DIMENSION", 0))
    if dimension <= 0:
        raise ValueError("DIMENSION not specified or invalid in VRP file.")

    if len(sections['depot']) == 0:
        raise ValueError("Missing DEPOT_SECTION in VRP file.")
    depot = []
    for l in sections['depot']:
        for elem in l.strip().split():
            num = int(elem)
            if num != -1:
                depot.append(num)
    if not depot:
        raise ValueError("No depot found in DEPOT_SECTION.")

    id_list = list(range(1, dimension + 1))
    idx_map = {nid: idx for idx, nid in enumerate(id_list)}

    node_coordinates = [None] * dimension
    for line in sections['node_coordinates']:
        parts = line.strip().split()
        if len(parts) < 3:
            raise ValueError("Invalid node coordinate line")
        nid = int(parts[0])
        idx = idx_map[nid]
        node_coordinates[idx] = [float(parts[1]), float(parts[2])]
    if any(nc is None for nc in node_coordinates):
        raise ValueError("Some node coordinates are missing.")

    demands = [None] * dimension
    for line in sections['demands']:
        parts = line.strip().split()
        if len(parts) < 2:
            raise ValueError("Invalid demand line")
        nid = int(parts[0])
        idx = idx_map[nid]
        demands[idx] = float(parts[1])
    if any(x is None for x in demands):
        raise ValueError("Some demands are missing.")

    service_times = [None] * dimension
    for line in sections['service_times']:
        parts = line.strip().split()
        if len(parts) < 2:
            raise ValueError("Invalid service time line")
        nid = int(parts[0])
        idx = idx_map[nid]
        service_times[idx] = float(parts[1])
    if any(x is None for x in service_times):
        raise ValueError("Some service times are missing.")

    time_windows = [None] * dimension
    for line in sections['time_windows']:
        parts = line.strip().split()
        if len(parts) < 3:
            raise ValueError("Invalid time window line")
        nid = int(parts[0])
        idx = idx_map[nid]
        time_windows[idx] = [float(parts[1]), float(parts[2])]
    if any(x is None for x in time_windows):
        raise ValueError("Some time window values are missing.")

    cap = meta.get("CAPACITY")
    if cap is None:
        raise ValueError("CAPACITY is missing in the meta information")
    vehicle_capacity = [float(cap)]
    if 'DISTANCE_LIMIT' in meta:
        distance_limits = [float(meta['DISTANCE_LIMIT'])]
    else:
        distance_limits = []

    return {
        "depot": depot,
        "node_coordinates": node_coordinates,
        "demands": demands,
        "vehicle_capacity": vehicle_capacity,
        "service_times": service_times,
        "time_windows": time_windows,
        "distance_limits": distance_limits
    }

def get_node_id_maps(instance):
    n_nodes = len(instance['node_coordinates'])
    id_list = list(range(1, n_nodes + 1))
    idx_map = {nid: idx for idx, nid in enumerate(id_list)}
    return id_list, idx_map

def distance(coords):
    n = len(coords)
    dist_matrix = []
    for i in range(n):
        row = []
        xi, yi = coords[i]
        for j in range(n):
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            row.append(d)
        dist_matrix.append(row)
    return dist_matrix

def cost(route, dist_matrix):
    n_nodes = len(dist_matrix)
    id_list = list(range(1, n_nodes+1))
    idx_map = {nid: idx for idx, nid in enumerate(id_list)}
    total_cost = 0.0
    for i in range(len(route)-1):
        n1, n2 = route[i], route[i+1]
        idx1 = idx_map[n1]
        idx2 = idx_map[n2]
        total_cost += dist_matrix[idx1][idx2]
    return total_cost

def total_cost(routes, dist_matrix):
    c = 0.0
    for route in routes:
        c += cost(route, dist_matrix)
    return c

def initial(instance, dist_matrix):
    id_list, idx_map = get_node_id_maps(instance)
    depot_ids = list(instance['depot'])
    depot_id = depot_ids[0]
    node_count = len(instance['node_coordinates'])
    node_ids = list(range(1, node_count + 1))
    customers = set(node_ids) - set(depot_ids)
    unvisited = set(customers)
    demand = instance['demands']
    tw = instance['time_windows']
    st = instance['service_times']
    cap = instance['vehicle_capacity'][0]
    distance_limits = instance['distance_limits']
    dist_limit = float('inf')
    if len(distance_limits) > 0:
        dist_limit = distance_limits[0]
    routes = []
    while unvisited:
        route = [depot_id]
        route_demand = 0.0
        route_dist = 0.0
        curr_time = tw[idx_map[depot_id]][0]
        last_service_end = curr_time + st[idx_map[depot_id]]
        curr_node = depot_id
        available_customers = sorted(unvisited)
        while True:
            feasible_customers = []
            for c in available_customers:
                idx_c = idx_map[c]
                dmd = demand[idx_c]
                total_dmd = route_demand + dmd
                travel = dist_matrix[idx_map[curr_node]][idx_c]
                route_dist_candidate = route_dist + travel
                min_tw, max_tw = tw[idx_c]
                arrive = last_service_end + travel
                start_service = max(arrive, min_tw)
                finish_service = start_service + st[idx_c]
                if (total_dmd <= cap + 1e-8 and
                    route_dist_candidate <= dist_limit + 1e-8 and
                    start_service <= max_tw + 1e-8):
                    feasible_customers.append((travel, c, start_service, finish_service))
            if feasible_customers:
                depot_idx = idx_map[depot_id]
                depot_to_unvisited = [
                    (dist_matrix[depot_idx][idx_map[c]], c) for c in unvisited
                ]
                depot_to_unvisited.sort()
                if depot_to_unvisited:
                    depot_dist_to_nearest, nearest_unvisited = depot_to_unvisited[0]
                    feasible_customers.sort()
                    curr_dist_to_nearest, next_c, start_service_c, finish_service_c = feasible_customers[0]
                    if depot_dist_to_nearest < curr_dist_to_nearest - 1e-10 and len(route) > 1:
                        break
                travel, next_c, start_service_c, finish_service_c = feasible_customers[0]
                route.append(next_c)
                route_demand += demand[idx_map[next_c]]
                route_dist += dist_matrix[idx_map[curr_node]][idx_map[next_c]]
                curr_node = next_c
                last_service_end = finish_service_c
                unvisited.remove(next_c)
                available_customers = sorted(unvisited)
            else:
                break
        while len(route) > 1 and route[-1] in depot_ids:
            route = route[:-1]
        routes.append(route)
    for i in range(len(routes)):
        while len(routes[i]) > 1 and routes[i][-1] in depot_ids:
            routes[i] = routes[i][:-1]
    return routes

def destroy(instance, dist_matrix, solution, ratio):
    id_list, idx_map = get_node_id_maps(instance)
    depot_ids = set(instance['depot'])
    node_count = len(instance['node_coordinates'])
    all_customers = [i for i in id_list if i not in depot_ids]
    num_to_remove = max(1, int(len(all_customers) * ratio))

    # Copy solution deeply for safe manipulation
    destroyed_solution = [list(route) for route in solution]
    customers_in_solution = []
    for route in destroyed_solution:
        customers_in_solution.extend([n for n in route if n not in depot_ids])

    # Choose customers to remove randomly from those assigned in current solution
    available_customers_for_removal = list(set(customers_in_solution))
    random.shuffle(available_customers_for_removal)
    removed_nodes = available_customers_for_removal[:num_to_remove]

    # Remove the chosen customers from the destroyed_solution
    removed_set = set(removed_nodes)
    new_destroyed_solution = []
    for route in destroyed_solution:
        filtered = [n for n in route if n not in removed_set]
        # Remove trailing depots and all depot IDs other than first element
        if not filtered:
            continue
        r_clip = [filtered[0]]
        r_clip.extend(n for n in filtered[1:] if n not in depot_ids)
        while len(r_clip) > 1 and r_clip[-1] in depot_ids:
            r_clip = r_clip[:-1]
        if len(r_clip) > 1:
            new_destroyed_solution.append(r_clip)
    destroyed_solution = new_destroyed_solution

    # Ensure all non-depot nodes are either in destroyed_solution or in removed_nodes
    customers_now = set()
    for route in destroyed_solution:
        customers_now.update([n for n in route if n not in depot_ids])
    missing = set(all_customers) - removed_set - customers_now
    for node in missing:
        removed_nodes.append(node)
        removed_set.add(node)
    assert set(all_customers) == set(removed_nodes).union(customers_now)
    assert not any([n in depot_ids for n in removed_nodes])
    return removed_nodes, destroyed_solution

def insert(destroyed_solution, removed_nodes, instance, dist_matrix):
    solution = [list(r) for r in destroyed_solution]
    id_list, idx_map = get_node_id_maps(instance)
    depot_ids = set(instance['depot'])
    depot_id = next(iter(depot_ids))
    demand = instance['demands']
    tw = instance['time_windows']
    st = instance['service_times']
    cap = instance['vehicle_capacity'][0]
    dist_limits = instance['distance_limits']
    has_dist_limit = len(dist_limits) > 0
    dist_limit = dist_limits[0] if has_dist_limit else float('inf')

    def check_feasible_open(route):
        if len(route) < 2:
            return False
        if route[0] not in depot_ids:
            return False
        if route[-1] in depot_ids:
            return False
        if any(n in depot_ids for n in route[1:]):
            return False
        seen_customers = set()
        for n in route:
            if n not in depot_ids:
                if n in seen_customers:
                    return False
                seen_customers.add(n)
        total_demand = 0.0
        idx_depot = idx_map[depot_id]
        time = tw[idx_depot][0] + st[idx_depot]
        last_node = depot_id
        distance_total = 0.0
        for pos in range(1, len(route)):
            cur_node = route[pos]
            i_last = idx_map[last_node]
            i_cur = idx_map[cur_node]
            travel = dist_matrix[i_last][i_cur]
            arrive = time + travel
            window_start, window_end = tw[i_cur][0], tw[i_cur][1]
            begin_service = max(arrive, window_start)
            if begin_service > window_end + 1e-8:
                return False
            finish_service = begin_service + st[i_cur]
            time = finish_service
            distance_total += travel
            if cur_node not in depot_ids:
                total_demand += demand[i_cur]
                if total_demand > cap + 1e-8:
                    return False
            last_node = cur_node
        if distance_total > dist_limit + 1e-8:
            return False
        return True

    for ins_node in removed_nodes:
        idx_ins = idx_map[ins_node]
        best_delta = float('inf')
        best_route = None
        best_pos = None
        best_new_route = None
        best_new_route_delta = float('inf')
        inserted = False

        for r_idx, route in enumerate(solution):
            # Possible insert positions: after the depot and everywhere else up to appending after last customer
            base_route = list(route)
            # Only first node may be depot, ensure no depot elsewhere
            base_route = [base_route[0]] + [n for n in base_route[1:] if n not in depot_ids]
            for pos in range(1, len(base_route)+1):
                new_route = base_route[:pos] + [ins_node] + base_route[pos:]
                # Remove any depot IDs except at first (sanitize); never add ins_node at pos=0
                if new_route[0] not in depot_ids or (len(new_route) > 1 and any(n in depot_ids for n in new_route[1:])):
                    continue
                if new_route[-1] in depot_ids:
                    continue
                customers_set = set(n for n in new_route if n not in depot_ids)
                if len(customers_set) != len([n for n in new_route if n not in depot_ids]):
                    continue
                if not check_feasible_open(new_route):
                    continue
                old_cost = cost(route, dist_matrix)
                new_cost = cost(new_route, dist_matrix)
                delta = new_cost - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_route = r_idx
                    best_pos = pos
        new_r = [depot_id, ins_node]
        if check_feasible_open(new_r):
            newroute_delta = cost(new_r, dist_matrix)
            if newroute_delta < best_delta:
                best_new_route = new_r
                best_new_route_delta = newroute_delta
        if best_new_route is not None and best_new_route_delta <= best_delta:
            solution.append(best_new_route)
            inserted = True
        elif best_route is not None:
            base = solution[best_route]
            base = [base[0]] + [n for n in base[1:] if n not in depot_ids]
            solution[best_route] = base[:best_pos] + [ins_node] + base[best_pos:]
            inserted = True
        else:
            if check_feasible_open([depot_id, ins_node]):
                solution.append([depot_id, ins_node])
                inserted = True
            else:
                raise ValueError(f"Cannot insert node {ins_node} feasibly")
    # Post-process: ensure routes are valid open, depots only at position 0, never at end, no depots in body, no duplicates
    new_solution = []
    for route in solution:
        if not route:
            continue
        r = [route[0]] + [n for n in route[1:] if n not in depot_ids]
        seen_customers = set()
        r_unique = []
        for i, n in enumerate(r):
            if n in depot_ids:
                if i != 0:
                    continue
                r_unique.append(n)
            else:
                if n not in seen_customers:
                    r_unique.append(n)
                    seen_customers.add(n)
        while len(r_unique) > 1 and r_unique[-1] in depot_ids:
            r_unique = r_unique[:-1]
        if len(r_unique) > 1 and r_unique[0] in depot_ids and r_unique[-1] not in depot_ids:
            new_solution.append(r_unique)
    return new_solution

def validate(solution, instance, dist_matrix):
    id_list, idx_map = get_node_id_maps(instance)
    depot_ids = set(instance['depot'])
    depot_id = next(iter(depot_ids))
    node_count = len(instance['node_coordinates'])
    demand = instance['demands']
    tw = instance['time_windows']
    st = instance['service_times']
    cap = instance['vehicle_capacity'][0]
    if instance.get("distance_limits", []):
        dist_limit = instance['distance_limits'][0]
    else:
        dist_limit = float('inf')
    all_customers = set(id_list) - depot_ids
    visited = []
    for route in solution:
        for n in route:
            if n not in depot_ids:
                visited.append(n)
    vset = set(visited)
    if len(visited) != len(vset):
        print("Visit constraint violated: some customers visited more than once.")
        return False
    if vset != all_customers:
        missing = all_customers - vset
        extra = vset - all_customers
        if missing:
            print(f"Visit constraint violated: missing customers: {sorted(missing)}")
        if extra:
            print(f"Visit constraint violated: extra nodes visited (not customers): {sorted(extra)}")
        return False
    for r, route in enumerate(solution):
        if not route:
            print(f"Route {r} is empty.")
            return False
        if (route[0] not in depot_ids) or (route[-1] in depot_ids):
            print(f"Open Route constraint violated on route {r}: must start at depot and end at customer.")
            return False
        for i in range(1, len(route)):
            if route[i] in depot_ids:
                print(f"Open Route constraint violated on route {r}: depot node found after start.")
                return False
        total_demand = 0.0
        for n in route:
            if n not in depot_ids:
                total_demand += demand[idx_map[n]]
        if total_demand > cap + 1e-8:
            print(f"Capacity constraint violated on route {r}: {total_demand} > {cap}")
            return False
        dsum = 0.0
        for i in range(len(route)-1):
            dsum += dist_matrix[idx_map[route[i]]][idx_map[route[i+1]]]
        if dsum > dist_limit + 1e-8:
            print(f"Distance Limit constraint violated on route {r}: {dsum} > {dist_limit}")
            return False
        t_cur = tw[idx_map[route[0]]][0] + st[idx_map[route[0]]]
        for i in range(1, len(route)):
            n = route[i]
            nidx = idx_map[n]
            prev_n = route[i-1]
            prev_idx = idx_map[prev_n]
            travel = dist_matrix[prev_idx][nidx]
            arrive = t_cur + travel
            tw_start, tw_end = tw[nidx][0], tw[nidx][1]
            start_service = max(arrive, tw_start)
            if start_service > tw_end + 1e-8:
                print(f"Time Window constraint violated on route {r} node {n}: start_service {start_service} > window_end {tw_end}")
                return False
            t_cur = start_service + st[nidx]
    return True

validation = validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to .vrp file')
    parser.add_argument('--iteration', type=int, default=100, help='Number of iterations (default: 100)')
    args = parser.parse_args()
    path = args.path
    iteration = args.iteration

    instance = read_vrp(path)
    coords = instance['node_coordinates']
    dist_matrix = distance(coords)
    id_list, idx_map = get_node_id_maps(instance)
    current_solution = initial(instance, dist_matrix)
    if not validation(solution=current_solution, instance=instance, dist_matrix=dist_matrix):
        raise Exception("Initial solution is infeasible.")

    current_cost = total_cost(current_solution, dist_matrix)
    best_solution = [list(r) for r in current_solution]
    best_cost = current_cost

    for step in range(iteration):
        ratio = random.uniform(0.0, 0.2)
        removed_nodes, destroyed_solution = destroy(instance, dist_matrix, solution=current_solution, ratio=ratio)
        current_solution = insert(destroyed_solution, removed_nodes, instance, dist_matrix)
        if not validation(solution=current_solution, instance=instance, dist_matrix=dist_matrix):
            raise Exception(f"Infeasible solution encountered at iteration {step}.")
        current_cost = total_cost(current_solution, dist_matrix)
        if current_cost <= best_cost:
            best_solution = [list(r) for r in current_solution]
            best_cost = current_cost
        else:
            p = random.uniform(0.0, 1.0)
            denom = max(1.0, iteration - step + 1)
            threshold = math.exp(-(current_cost - best_cost) * iteration * 10 / denom)
            if p > threshold:
                current_solution = [list(r) for r in best_solution]
                current_cost = best_cost
    feasible_vehicle_routes = best_solution
    print(f"the process is successful, the best cost is {best_cost}")