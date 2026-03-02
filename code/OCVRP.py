import re
import math
import random
import copy
import argparse
import sys

def read_vrp(path: str):
    depot_ids = []
    node_coord_dict = {}
    demand_dict = {}
    dimension = None
    capacity = None
    coord_section = False
    demand_section = False
    depot_section = False

    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        l = line.strip()
        if not l or l == 'EOF':
            continue
        if l.startswith("DIMENSION"):
            dimension = int(re.split(r'\s*:\s*', l)[1])
        elif l.startswith("CAPACITY"):
            capacity = float(re.split(r'\s*:\s*', l)[1])
        elif l.startswith("NODE_COORD_SECTION"):
            coord_section = True
            demand_section = False
            depot_section = False
            continue
        elif l.startswith("DEMAND_SECTION"):
            coord_section = False
            demand_section = True
            depot_section = False
            continue
        elif l.startswith("DEPOT_SECTION"):
            coord_section = False
            demand_section = False
            depot_section = True
            continue

        if coord_section:
            splits = l.split()
            if len(splits) >= 3:
                idx = int(splits[0])
                x = float(splits[1])
                y = float(splits[2])
                node_coord_dict[idx] = [x, y]
        elif demand_section:
            splits = l.split()
            if len(splits) >= 2:
                idx = int(splits[0])
                dem = float(splits[1])
                demand_dict[idx] = dem
        elif depot_section and l and l != '-1':
            depot_ids.append(int(l))

    if dimension is None or capacity is None:
        raise ValueError("DIMENSION or CAPACITY missing.")
    if not depot_ids:
        raise ValueError("DEPOT_SECTION missing depot node.")
    if sorted(node_coord_dict.keys()) != sorted(demand_dict.keys()):
        raise ValueError("Mismatch in node coordinates and demand ids.")
    all_node_ids = sorted(node_coord_dict.keys())
    depot_id_in_file = depot_ids[0]
    customer_ids = [i for i in all_node_ids if i != depot_id_in_file]
    ordered_ids = [depot_id_in_file] + customer_ids

    node_id_to_new_idx = {ordered_ids[i]: i for i in range(len(ordered_ids))}
    node_coordinates = [node_coord_dict[i] for i in ordered_ids]
    demands = [demand_dict[i] for i in ordered_ids]
    depot_zero_based = [0]
    vehicle_capacity = [capacity]

    return {
        "depot": depot_zero_based,
        "node_coordinates": node_coordinates,
        "demands": demands,
        "vehicle_capacity": vehicle_capacity
    }

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
    if len(route) == 0 or (isinstance(route[0], int) and len(route) <= 1):
        return 0.0
    if any(isinstance(r, list) for r in route):
        total = 0.0
        for r in route:
            total += cost(r, dist_matrix)
        return total
    if len(route) <= 1:
        return 0.0
    total_cost = 0.0
    for i in range(len(route) - 1):
        from_idx = route[i]
        to_idx = route[i+1]
        total_cost += dist_matrix[from_idx][to_idx]
    return total_cost

def initial(instance, dist_matrix):
    depot_idx = 0
    n = len(instance["node_coordinates"])
    demands = instance["demands"]
    vehicle_capacity = float(instance["vehicle_capacity"][0])
    unvisited = set(range(1, n))
    routes = []

    while unvisited:
        route = [depot_idx]
        capacity_left = vehicle_capacity
        current_pos = depot_idx

        while True:
            # Only non-depot nodes
            feasible = [i for i in unvisited if demands[i] <= capacity_left]
            if not feasible:
                break
            nearest_from_current = min(feasible, key=lambda j: dist_matrix[current_pos][j])
            dist_current_to_nearest = dist_matrix[current_pos][nearest_from_current]
            nearest_from_depot = min(feasible, key=lambda j: dist_matrix[depot_idx][j])
            dist_depot_to_nearest = dist_matrix[depot_idx][nearest_from_depot]
            if route != [depot_idx] and dist_depot_to_nearest < dist_current_to_nearest:
                break
            nearest_cust = nearest_from_current
            route.append(nearest_cust)
            unvisited.remove(nearest_cust)
            capacity_left -= demands[nearest_cust]
            current_pos = nearest_cust
        routes.append(route)
    # Ensure that every route starts at depot, contains only customers after, depot is not in any other position
    cleaned_routes = []
    for r in routes:
        if r and r[0] == depot_idx:
            cleaned_routes.append([r[0]] + [x for x in r[1:] if x != depot_idx])
    return cleaned_routes

def destroy(instance, dist_matrix, solution, ratio):
    depot_idx = 0
    n = len(instance["node_coordinates"])
    customers = set()
    for route in solution:
        for i, node in enumerate(route):
            if node != depot_idx:
                customers.add(node)
    num_to_remove = max(1, int(len(customers) * ratio))
    removed_nodes = []
    destroyed_solution = copy.deepcopy(solution)
    customer_indices = [i for i in range(1, n)]
    center = random.choice([i for i in customers])
    neighbor_candidates = sorted([i for i in customer_indices if i in customers and i != center],
                                 key=lambda j: dist_matrix[center][j])
    neighbor_order = [center] + neighbor_candidates
    destroyed_in_route = set()
    route_idx_containing = {}
    for r_idx, route in enumerate(destroyed_solution):
        for node in route:
            if node != depot_idx:
                route_idx_containing[node] = r_idx
    candidate_set = set(customers)
    remove_cnt = 0
    marked_removed = set()
    destroyed_route_indices = set()
    for neighbor in neighbor_order:
        if remove_cnt >= num_to_remove:
            break
        if neighbor not in candidate_set or neighbor in marked_removed:
            continue
        route_idx = route_idx_containing[neighbor]
        if route_idx in destroyed_route_indices:
            continue
        route = destroyed_solution[route_idx]
        pos_list = [idx for idx, node in enumerate(route) if node != depot_idx]
        if not pos_list:
            continue
        neighbor_pos = None
        for p in pos_list:
            if route[p] == neighbor:
                neighbor_pos = p
                break
        if neighbor_pos is None:
            continue
        available_len = len(pos_list)
        max_block = min(available_len, num_to_remove - remove_cnt)
        block_length = random.randint(1, max_block)
        block_starts = []
        for start in range(neighbor_pos - block_length + 1, neighbor_pos + 1):
            end = start + block_length - 1
            if start >= min(pos_list) and end <= max(pos_list):
                block_starts.append(start)
        if not block_starts:
            block_start = neighbor_pos
            block_length = 1
        else:
            block_start = random.choice(block_starts)
        block_indices = list(range(block_start, block_start + block_length))
        nodes_to_remove = [route[i] for i in block_indices if route[i] != depot_idx]
        destroyed_solution[route_idx] = [node for idx, node in enumerate(route) if idx not in block_indices]
        removed_nodes.extend(nodes_to_remove)
        remove_cnt += len(nodes_to_remove)
        destroyed_route_indices.add(route_idx)
        marked_removed.update(nodes_to_remove)
    cleaned_solution = []
    for route in destroyed_solution:
        if any(node != depot_idx for node in route):
            # Remove depot if accidentally left at non-start positions
            cleaned_solution.append([depot_idx] + [n for n in route[1:] if n != depot_idx])
    destroyed_solution = cleaned_solution
    all_nodes_after = set()
    for route in destroyed_solution:
        for i, n in enumerate(route):
            if n != depot_idx:
                all_nodes_after.add(n)
    assert depot_idx not in removed_nodes
    assert set(removed_nodes).isdisjoint(all_nodes_after)
    custs_before = set(customers)
    assert custs_before == (set(removed_nodes).union(all_nodes_after))
    return removed_nodes, destroyed_solution

def insert(destroyed_solution, removed_nodes, instance, dist_matrix):
    depot_id = instance["depot"][0]
    n = len(instance["node_coordinates"])
    demands = instance["demands"]
    vehicle_capacity = float(instance["vehicle_capacity"][0])
    routes = copy.deepcopy(destroyed_solution)
    assigned_customers = set()
    for route in routes:
        assigned_customers.update([node for node in route[1:] if node != depot_id])
    depot_idx = depot_id

    for node_id in removed_nodes:
        node_idx = node_id
        demand = demands[node_idx]
        min_cost_increase = None
        best_route_idx = None
        best_insert_pos = None

        for r_idx, route in enumerate(routes):
            route_customers = [x for x in route[1:] if x != depot_idx]
            cap_used = sum(demands[x] for x in route_customers)
            if cap_used + demand > vehicle_capacity:
                continue
            for ins_pos in range(1, len(route)+1):
                if node_idx in route:
                    continue
                prev_node = route[ins_pos-1]
                if ins_pos < len(route):
                    next_node = route[ins_pos]
                    delta = (
                        dist_matrix[prev_node][node_idx] + dist_matrix[node_idx][next_node]
                        - dist_matrix[prev_node][next_node]
                    )
                else:
                    delta = dist_matrix[prev_node][node_idx]
                if (min_cost_increase is None) or (delta < min_cost_increase):
                    min_cost_increase = delta
                    best_route_idx = r_idx
                    best_insert_pos = ins_pos

        if demand <= vehicle_capacity:
            cost_new_route = dist_matrix[depot_idx][node_idx]
            if (min_cost_increase is None) or (cost_new_route < min_cost_increase):
                min_cost_increase = cost_new_route
                best_route_idx = None
                best_insert_pos = None

        if best_route_idx is not None:
            route = routes[best_route_idx]
            route = route[:best_insert_pos] + [node_idx] + route[best_insert_pos:]
            # Remove any depot not at start
            cleaned = [depot_idx] + [x for x in route[1:] if x != depot_idx]
            routes[best_route_idx] = cleaned
        else:
            routes.append([depot_idx, node_idx])

        assigned_customers.add(node_idx)

    final_routes = []
    for route in routes:
        # Only first position can be depot, enforce this
        cleaned = [route[0]] + [x for x in route[1:] if x != depot_idx]
        route_customers = [x for x in cleaned[1:] if x != depot_idx]
        if len(route_customers) > 0:
            final_routes.append([cleaned[0]] + route_customers)
    all_assigned_customers = set()
    for route in final_routes:
        for node in route[1:]:
            if node != depot_idx:
                all_assigned_customers.add(node)
    assert all(node != depot_idx for node in removed_nodes)
    assert set(removed_nodes).issubset(all_assigned_customers)
    assert len(all_assigned_customers) == len(set(all_assigned_customers))
    return final_routes

def validate(solution, instance, dist_matrix):
    depot_id = instance['depot'][0]
    n = len(instance['node_coordinates'])
    demands = instance['demands']
    vehicle_capacity = float(instance['vehicle_capacity'][0])
    all_customers = set(range(1, n))
    visited_customers = []
    customer_occurrences = {}

    for route in solution:
        if not route:
            print("Route is empty.")
            return False
        # Only depot may appear at the start
        if route[0] != depot_id:
            print(f"Open route constraint violated: Route does not start at depot. Route: {route}")
            return False
        for i, node in enumerate(route):
            if i == 0:
                if node != depot_id:
                    print(f"Start of route is not depot: {route}")
                    return False
            else:
                if node == depot_id:
                    print(f"Depot appears at non-initial position in route: {route}")
                    return False
            if node == depot_id:
                continue
            if node < 1 or node >= n:
                print("Node id out of allowed range (not in customers)", node)
                return False
            visited_customers.append(node)
            if node in customer_occurrences:
                customer_occurrences[node] += 1
            else:
                customer_occurrences[node] = 1

    for c in all_customers:
        if customer_occurrences.get(c, 0) > 1:
            print(f"Visit constraint violated: Customer {c} visited more than once.")
            return False
        if customer_occurrences.get(c, 0) == 0:
            print(f"Visit constraint violated: Customer {c} not visited.")
            return False
    if len(visited_customers) != len(set(visited_customers)):
        print("Visit constraint violated: Some customers are visited more than once.")
        return False
    if set(visited_customers) != all_customers:
        print("Visit constraint violated: Not all customers are visited exactly once.")
        return False

    for idx, route in enumerate(solution):
        cap_used = 0.0
        for node in route:
            if node == depot_id:
                continue
            cap_used += demands[node]
        if cap_used > vehicle_capacity + 1e-8:
            print(f"Capacity constraint violated: Route {idx} uses {cap_used} > vehicle capacity {vehicle_capacity}")
            return False

        if len(route) == 0:
            print(f"Route {idx} is empty.")
            return False
        if route[0] != depot_id:
            print(f"Open route constraint violated: Route {idx} does not start at depot.")
            return False

    return True

validation = validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Open CVRP solver')
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
    if not validation(solution=current_solution, instance=instance, dist_matrix=dist_matrix):
        raise Exception("Initial solution is invalid")
    current_cost = cost(route=current_solution, dist_matrix=dist_matrix)
    best_solution = copy.deepcopy(current_solution)
    best_cost = current_cost

    # SA parameters
    T0 = 1.0
    Tmin = 1e-4
    alpha = 0.995
    T = T0

    for step in range(iteration):
        ratio = random.uniform(0.01, 0.2)
        removed_nodes, destroyed_solution = destroy(instance, dist_matrix, solution=current_solution, ratio=ratio)
        candidate_solution = insert(destroyed_solution, removed_nodes, instance, dist_matrix)
        if not validation(solution=candidate_solution, instance=instance, dist_matrix=dist_matrix):
            raise Exception("Solution after iteration {} is invalid".format(step))
        candidate_cost = cost(route=candidate_solution, dist_matrix=dist_matrix)
        delta = candidate_cost - current_cost

        if delta < 0:
            current_solution = candidate_solution
            current_cost = candidate_cost
            if candidate_cost < best_cost:
                best_solution = copy.deepcopy(candidate_solution)
                best_cost = candidate_cost
        else:
            prob = math.exp(-delta / T) if T > 0 else 0.0
            if random.uniform(0, 1) < prob:
                current_solution = candidate_solution
                current_cost = candidate_cost
            else:
                # revert to best_solution
                current_solution = copy.deepcopy(best_solution)
                current_cost = best_cost

        T = max(Tmin, T * alpha)

    print(f'the process is successful, the best cost is {best_cost}')