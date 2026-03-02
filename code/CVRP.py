import re
import math
import random
import copy
import argparse
import sys

def read_vrp(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()
    node_coord_dict = {}
    demand_dict = {}
    depot_ids = []
    dimension = None
    capacity = None
    in_node_coord_section = False
    in_demand_section = False
    in_depot_section = False
    for line in lines:
        line = line.strip()
        if not line or line == 'EOF':
            continue
        if line.startswith('DIMENSION'):
            dimension = int(re.search(r':\s*(\d+)', line).group(1))
            continue
        if line.startswith('CAPACITY'):
            capacity = float(re.search(r':\s*([\d.]+)', line).group(1))
            continue
        if line == 'NODE_COORD_SECTION':
            in_node_coord_section = True
            in_demand_section = False
            in_depot_section = False
            continue
        if line == 'DEMAND_SECTION':
            in_node_coord_section = False
            in_demand_section = True
            in_depot_section = False
            continue
        if line == 'DEPOT_SECTION':
            in_node_coord_section = False
            in_demand_section = False
            in_depot_section = True
            continue
        if in_node_coord_section:
            tokens = line.split()
            idx = int(tokens[0])
            x = float(tokens[1])
            y = float(tokens[2])
            node_coord_dict[idx] = [x, y]
        elif in_demand_section:
            tokens = line.split()
            idx = int(tokens[0])
            d = float(tokens[1])
            demand_dict[idx] = d
        elif in_depot_section:
            val = int(line)
            if val == -1:
                continue
            depot_ids.append(val)
    if not depot_ids:
        raise ValueError('Depot information missing in VRP file.')
    if dimension is None:
        raise ValueError('DIMENSION missing in VRP file.')
    if capacity is None:
        raise ValueError('CAPACITY missing in VRP file.')
    all_ids = sorted(node_coord_dict.keys())
    if len(all_ids) != dimension:
        raise ValueError('Parsed node_coord_dict length does not match DIMENSION.')
    if len(demand_dict) != dimension:
        raise ValueError('Parsed demand_dict length does not match DIMENSION.')
    depot_id = depot_ids[0]
    if depot_id not in all_ids:
        raise ValueError('Depot id not present in node_coord_dict.')
    customer_ids = [nid for nid in all_ids if nid != depot_id]
    node_id_list = [depot_id] + sorted(customer_ids)
    node_coordinates = [node_coord_dict[nid] for nid in node_id_list]
    demands = [demand_dict[nid] for nid in node_id_list]
    return {
        "depot": [depot_id],
        "node_coordinates": node_coordinates,
        "demands": demands,
        "vehicle_capacity": [capacity]
    }

def _id_index_maps(instance):
    depot_id = instance["depot"][0]
    n_nodes = len(instance["node_coordinates"])
    n_coords = len(instance["node_coordinates"])
    possible_ids = set(range(1, n_coords+1))
    customers = sorted([nid for nid in possible_ids if nid != depot_id])
    node_id_list = [depot_id] + customers
    id_to_idx = {nid: idx for idx, nid in enumerate(node_id_list)}
    idx_to_id = {idx: nid for idx, nid in enumerate(node_id_list)}
    return id_to_idx, idx_to_id

def distance(coords):
    n = len(coords)
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xa, ya = coords[i]
        for j in range(n):
            xb, yb = coords[j]
            dist_matrix[i][j] = math.hypot(xa - xb, ya - yb)
    return dist_matrix

def cost(route, dist_matrix):
    if not route or len(route) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(route[:-1], route[1:]):
        total += dist_matrix[a][b]
    return total

def initial(instance, dist_matrix):
    id_to_idx, idx_to_id = _id_index_maps(instance)
    depot_id = instance['depot'][0]
    n_nodes = len(instance['node_coordinates'])
    possible_ids = set(range(1, n_nodes+1))
    customers = [nid for nid in sorted(possible_ids) if nid != depot_id]
    unvisited = set(customers)
    routes = []
    capacity = instance['vehicle_capacity'][0]
    demands = instance['demands']
    while unvisited:
        route_ids = [depot_id]
        load = 0.0
        curr_id = depot_id
        curr_idx = id_to_idx[curr_id]
        while True:
            candidates = []
            min_dist = float('inf')
            for u in sorted(unvisited):
                u_idx = id_to_idx[u]
                if load + demands[u_idx] > capacity + 1e-8:
                    continue
                d = dist_matrix[curr_idx][u_idx]
                if d < min_dist - 1e-12:
                    min_dist = d
                    candidates = [u]
                elif abs(d - min_dist) < 1e-12:
                    candidates.append(u)
            if not candidates:
                break
            nearest = min(candidates)
            route_ids.append(nearest)
            load += demands[id_to_idx[nearest]]
            curr_id = nearest
            curr_idx = id_to_idx[curr_id]
            unvisited.remove(nearest)
        route_ids.append(depot_id)
        route = [id_to_idx[nid] for nid in route_ids]
        routes.append(route)
    return routes

def destroy(instance, dist_matrix, solution, ratio):
    id_to_idx, idx_to_id = _id_index_maps(instance)
    depot_id = instance['depot'][0]
    n_nodes = len(instance['node_coordinates'])
    possible_ids = set(range(1, n_nodes+1))
    customers = [nid for nid in sorted(possible_ids) if nid != depot_id]
    n_customers = len(customers)
    if n_customers == 0:
        return [], copy.deepcopy(solution)
    k_to_remove = int(round(n_customers * ratio))
    k_to_remove = max(1, min(k_to_remove, n_customers))
    center = random.choice(customers)
    center_idx = id_to_idx[center]
    neighbor_dists = []
    for nid in customers:
        if nid == center:
            continue
        idx = id_to_idx[nid]
        neighbor_dists.append((dist_matrix[center_idx][idx], nid))
    neighbor_dists.sort()
    candidates = [center] + [nid for _, nid in neighbor_dists]
    destroyed_nodes = []
    destroyed_node_set = set()
    used_route = set()
    remaining_to_remove = k_to_remove
    cust_to_route = {}
    for r_idx, route in enumerate(solution):
        for pos in range(1, len(route)-1):
            nid = idx_to_id[route[pos]]
            if nid != depot_id:
                cust_to_route[nid] = (r_idx, pos)
    for cand in candidates:
        if remaining_to_remove <= 0:
            break
        if cand in destroyed_node_set:
            continue
        if cand not in cust_to_route:
            continue
        r_idx, pos = cust_to_route[cand]
        if r_idx in used_route:
            continue
        route = solution[r_idx]
        positions = [i for i in range(1, len(route)-1) if idx_to_id[route[i]] != depot_id and idx_to_id[route[i]] not in destroyed_node_set]
        if not positions or pos not in positions:
            used_route.add(r_idx)
            continue
        available_customers = [idx_to_id[route[p]] for p in positions]
        cand_loc = positions.index(pos)
        max_win = min(len(available_customers), remaining_to_remove)
        win_size = random.randint(1, max_win)
        max_left = cand_loc
        max_right = len(available_customers) - cand_loc - 1
        left_ext = random.randint(0, min(max_left, win_size - 1))
        right_ext = min(win_size - 1 - left_ext, max_right)
        left = cand_loc - left_ext
        right = cand_loc + right_ext
        if right >= len(available_customers):
            shift = right - len(available_customers) + 1
            left = max(0, left - shift)
            right = left + win_size - 1
        ids_to_remove = available_customers[left:right+1]
        ids_to_remove = [nid for nid in ids_to_remove if nid != depot_id and nid not in destroyed_node_set]
        if len(ids_to_remove) == 0:
            used_route.add(r_idx)
            continue
        if len(ids_to_remove) > remaining_to_remove:
            ids_to_remove = ids_to_remove[:remaining_to_remove]
        destroyed_nodes.extend(ids_to_remove)
        destroyed_node_set.update(ids_to_remove)
        used_route.add(r_idx)
        remaining_to_remove -= len(ids_to_remove)
    destroyed_nodes = destroyed_nodes[:k_to_remove]
    to_remove_set = set(destroyed_nodes)
    new_solution = []
    for route in solution:
        cleaned = []
        for idx in route:
            nid = idx_to_id[idx]
            if nid != depot_id and nid in to_remove_set:
                continue
            cleaned.append(idx)
        merged = []
        for idx in cleaned:
            if not merged:
                merged.append(idx)
            elif idx == id_to_idx[depot_id] and merged[-1] == id_to_idx[depot_id]:
                continue
            else:
                merged.append(idx)
        inner_customers = [idx_to_id[idx] for idx in merged if idx_to_id[idx] != depot_id]
        if len(inner_customers) == 0:
            continue
        if merged[0] == id_to_idx[depot_id] and merged[-1] == id_to_idx[depot_id]:
            new_solution.append(merged)
    contained = set()
    for route in new_solution:
        contained.update([idx_to_id[idx] for idx in route if idx_to_id[idx] != depot_id])
    assert depot_id not in destroyed_nodes
    assert set(destroyed_nodes).isdisjoint(contained)
    assert set(destroyed_nodes).union(contained) == set(customers)
    assert 1 <= len(destroyed_nodes) <= k_to_remove
    return destroyed_nodes, new_solution

def insert(destroyed_solution, removed_nodes, instance, dist_matrix):
    id_to_idx, idx_to_id = _id_index_maps(instance)
    depot_id = instance["depot"][0]
    capacity = instance["vehicle_capacity"][0]
    demands = instance["demands"]
    solution = copy.deepcopy(destroyed_solution)
    routes_loads = []
    for route in solution:
        routes_loads.append(sum(demands[idx] for idx in route if idx_to_id[idx] != depot_id))
    for nid in removed_nodes:
        if nid == depot_id:
            continue
        demand = demands[id_to_idx[nid]]
        best_cost_increase = float('inf')
        best_route_idx = None
        best_pos = None
        for r_idx, route in enumerate(solution):
            load = routes_loads[r_idx]
            for pos in range(1, len(route)):
                if load + demand > capacity + 1e-8:
                    continue
                u = route[pos-1]
                v = route[pos]
                nidx = id_to_idx[nid]
                cost_inc = dist_matrix[u][nidx] + dist_matrix[nidx][v] - dist_matrix[u][v]
                if cost_inc < best_cost_increase:
                    best_cost_increase = cost_inc
                    best_route_idx = r_idx
                    best_pos = pos
        if best_route_idx is not None:
            nidx = id_to_idx[nid]
            solution[best_route_idx].insert(best_pos, nidx)
            routes_loads[best_route_idx] += demand
        else:
            nidx = id_to_idx[nid]
            solution.append([id_to_idx[depot_id], nidx, id_to_idx[depot_id]])
            routes_loads.append(demand)
    new_routes = []
    for route in solution:
        inner_customers = [idx_to_id[idx] for idx in route if idx_to_id[idx] != depot_id]
        if len(inner_customers) == 0:
            continue
        merged = []
        for idx in route:
            if len(merged) == 0:
                merged.append(idx)
            elif idx == id_to_idx[depot_id] and merged[-1] == id_to_idx[depot_id]:
                continue
            else:
                merged.append(idx)
        if len(merged) >= 2 and merged[0] == id_to_idx[depot_id] and merged[-1] == id_to_idx[depot_id] and any(idx_to_id[idx] != depot_id for idx in merged):
            new_routes.append(merged)
    for r in new_routes:
        assert r[0] == id_to_idx[depot_id] and r[-1] == id_to_idx[depot_id]
        assert id_to_idx[depot_id] not in r[1:-1]
        assert all(idx!=id_to_idx[depot_id] for idx in r[1:-1])
    return new_routes

def validate(solution, instance, dist_matrix):
    id_to_idx, idx_to_id = _id_index_maps(instance)
    depot_id = instance['depot'][0]
    n_nodes = len(instance['node_coordinates'])
    possible_ids = set(range(1, n_nodes+1))
    customers = set(nid for nid in possible_ids if nid != depot_id)
    vehicle_capacity = instance['vehicle_capacity'][0]
    demands = instance['demands']
    for ridx, route in enumerate(solution):
        load = 0.0
        for idx in route:
            nid = idx_to_id[idx]
            if nid == depot_id:
                continue
            load += demands[id_to_idx[nid]]
        if load > vehicle_capacity + 1e-8:
            print(f"Capacity constraint violated in route {ridx}: load {load} > capacity {vehicle_capacity}")
            return False
    cust_visited = []
    for route in solution:
        cust_visited.extend([idx_to_id[idx] for idx in route if idx_to_id[idx] != depot_id])
    cust_visited_multiset = set([nid for nid in cust_visited if cust_visited.count(nid) > 1])
    if len(cust_visited) != len(set(cust_visited)):
        print(f"Visit constraint violated: customers visited more than once:", sorted(cust_visited_multiset))
        return False
    if set(cust_visited) != customers:
        missing = customers.difference(set(cust_visited))
        extra = set(cust_visited).difference(customers)
        if missing:
            print("Visit constraint violated: missing customers", sorted(missing))
        if extra:
            print("Visit constraint violated: invalid customer IDs in solution", sorted(extra))
        return False
    for ridx, route in enumerate(solution):
        if not route or route[0] != id_to_idx[depot_id] or route[-1] != id_to_idx[depot_id]:
            print(f"Depot constraint violated in route {ridx}: route must start and end with depot as index {id_to_idx[depot_id]}")
            return False
        for i in range(1, len(route)):
            if route[i] == id_to_idx[depot_id] and route[i-1] == id_to_idx[depot_id]:
                print(f"Route {ridx}: consecutive depots found, invalid.")
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
    if not validate(current_solution, instance, dist_matrix):
        raise Exception("Initial solution not feasible. Exiting.")
    current_cost = sum([cost(r, dist_matrix) for r in current_solution])
    best_solution = copy.deepcopy(current_solution)
    best_cost = current_cost
    for step in range(iteration):
        ratio = random.uniform(0, 0.2)
        removed_nodes, destroyed_solution = destroy(instance, dist_matrix, current_solution, ratio)
        current_solution = insert(destroyed_solution, removed_nodes, instance, dist_matrix)
        if not validate(current_solution, instance, dist_matrix):
            raise Exception("Solution after insertion not feasible. Exiting.")
        current_cost = sum([cost(r, dist_matrix) for r in current_solution])
        if current_cost <= best_cost:
            best_solution = copy.deepcopy(current_solution)
            best_cost = current_cost
        else:
            p = random.uniform(0,1)
            denom = max(1e-8, (iteration-step+1))
            exponent = -(current_cost - best_cost) * iteration * 10 / denom
            threshold = math.exp(exponent) if exponent > -700 else 0.0
            if p > threshold:
                current_solution = copy.deepcopy(best_solution)
                current_cost = best_cost
    print(f"the process is successful, the best cost is {best_cost}")