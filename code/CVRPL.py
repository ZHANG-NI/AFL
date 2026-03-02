import re
import math
import random
import copy
import argparse
import sys

def read_vrp(path: str):
    def is_section_header(line):
        return re.match(r'^[A-Z_]+_SECTION', line.strip())
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != '']
    depot = []
    node_coordinates = []
    demands = []
    capacity = []
    distance_limit = []
    n_nodes = None
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith('DIMENSION'):
            n_nodes = int(line.split(':')[1])
            idx += 1
        elif line.startswith('CAPACITY'):
            cap = float(line.split(':')[1])
            capacity.append(cap)
            idx += 1
        elif line.startswith('DISTANCE_LIMIT'):
            limit = float(line.split(':')[1])
            distance_limit.append(limit)
            idx += 1
        elif line.startswith('DEPOT_SECTION'):
            idx += 1
            while idx < len(lines):
                depot_line = lines[idx]
                if depot_line == '-1' or depot_line == 'EOF' or is_section_header(depot_line):
                    break
                if depot_line.isdigit():
                    depot.append(int(depot_line))
                idx += 1
            while idx < len(lines) and (lines[idx] == '-1' or lines[idx] == 'EOF'):
                idx += 1
        elif line.startswith('NODE_COORD_SECTION'):
            idx += 1
            node_coordinates = []
            while idx < len(lines):
                coord_line = lines[idx]
                if coord_line == '-1' or coord_line == 'EOF' or is_section_header(coord_line):
                    break
                tokens = coord_line.split()
                if len(tokens) >= 3:
                    node_coordinates.append([float(tokens[1]), float(tokens[2])])
                idx += 1
            while idx < len(lines) and (lines[idx] == '-1' or lines[idx] == 'EOF'):
                idx += 1
        elif line.startswith('DEMAND_SECTION'):
            idx += 1
            demands = []
            while idx < len(lines):
                demand_line = lines[idx]
                if demand_line == '-1' or demand_line == 'EOF' or is_section_header(demand_line):
                    break
                tokens = demand_line.split()
                if len(tokens) >= 2:
                    demands.append(float(tokens[1]))
                idx += 1
            while idx < len(lines) and (lines[idx] == '-1' or lines[idx] == 'EOF'):
                idx += 1
        else:
            idx += 1
    return {
        'depot': depot,
        'node_coordinates': node_coordinates,
        'demands': demands,
        'capacity': capacity,
        'distance_limit': distance_limit
    }

def distance(coords):
    n = len(coords)
    dist_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            row.append(math.hypot(dx, dy))
        dist_matrix.append(row)
    return dist_matrix

def cost(route, dist_matrix):
    total_cost = 0.0
    for single_route in route:
        for i in range(len(single_route) - 1):
            from_idx = single_route[i] - 1
            to_idx = single_route[i + 1] - 1
            total_cost += dist_matrix[from_idx][to_idx]
    return total_cost

def initial(instance, dist_matrix):
    depot = instance['depot'][0]
    node_count = len(instance['node_coordinates'])
    capacity = instance['capacity'][0]
    distance_limit = instance['distance_limit'][0]
    demands = instance['demands']
    customers = [i + 1 for i, d in enumerate(demands) if d > 0]
    unvisited = set(customers)
    routes = []
    while unvisited:
        route = [depot]
        current_load = 0.0
        traveled = 0.0
        last = depot
        while unvisited:
            candidate = None
            best_extra = float('inf')
            for cust in unvisited:
                demand = demands[cust - 1]
                if current_load + demand - 1e-8 > capacity:
                    continue
                extra = dist_matrix[last - 1][cust - 1]
                return_dist = dist_matrix[cust - 1][depot - 1]
                if traveled + extra + return_dist - 1e-8 > distance_limit:
                    continue
                if extra < best_extra:
                    best_extra = extra
                    candidate = cust
            if candidate is not None:
                route.append(candidate)
                unvisited.remove(candidate)
                current_load += demands[candidate - 1]
                traveled += dist_matrix[last - 1][candidate - 1]
                last = candidate
            else:
                break
        traveled += dist_matrix[last - 1][depot - 1]
        route.append(depot)
        routes.append(route)
    # Remove empty (no customers) routes
    clean_routes = []
    depot_set = set(instance['depot'])
    for r in routes:
        customer_nodes = [n for n in r if n not in depot_set]
        if len(customer_nodes) > 0:
            clean_routes.append(r)
    # Customer uniqueness check
    all_customers = [i + 1 for i, d in enumerate(demands) if d > 0]
    customers_in_solution = []
    for r in clean_routes:
        customers_in_solution += [n for n in r if n not in depot_set]
    assert sorted(customers_in_solution) == sorted(all_customers)
    return clean_routes

def destroy(instance, dist_matrix, solution, ratio):
    depot_set = set(instance['depot']) if 'depot' in instance else set()
    stations = set()
    demands = instance['demands']
    node_count = len(demands)
    all_customers = [i + 1 for i in range(node_count) if (demands[i] > 0 and (i + 1) not in depot_set and (i + 1) not in stations)]
    n_remove = int(len(all_customers) * ratio)
    if n_remove < 1:
        n_remove = 1
    if n_remove > len(all_customers):
        n_remove = len(all_customers)
    removed_nodes = set()
    destroyed_solution = [route[:] for route in solution]
    customers_left = set()
    for route in destroyed_solution:
        customers_left.update([n for n in route if n not in depot_set])
    customer_list = list(customers_left)
    random.shuffle(customer_list)
    center = random.choice(customer_list)
    dist_to_center = []
    for cust in all_customers:
        if cust == center:
            dist = 0.0
        else:
            dist = dist_matrix[center - 1][cust - 1]
        dist_to_center.append((dist, cust))
    dist_to_center.sort()
    ordered_candidates = [center] + [cust for (d, cust) in dist_to_center if cust != center]
    idx = 0
    while len(removed_nodes) < n_remove and idx < len(ordered_candidates):
        node = ordered_candidates[idx]
        found = False
        for ridx, route in enumerate(destroyed_solution):
            if node in route:
                # Remove only if not already removed
                inner = [n for n in route if n not in depot_set]
                if node in inner and node not in removed_nodes:
                    for pos in range(1, len(route) - 1):
                        if route[pos] == node:
                            del route[pos]
                            removed_nodes.add(node)
                            found = True
                            break
                    break
        idx += 1
    # Ensure removed_nodes and remaining customers are disjoint
    for route in destroyed_solution:
        for node in route:
            assert node not in removed_nodes or node in depot_set
    # Clean up redundant depot repeats and empty routes
    for idx, route in enumerate(destroyed_solution):
        cleaned_route = []
        prev = None
        for n in route:
            if n in depot_set:
                if prev != n:
                    cleaned_route.append(n)
            else:
                cleaned_route.append(n)
            prev = n
        destroyed_solution[idx] = cleaned_route
    destroyed_solution = [
        r for r in destroyed_solution if any(n not in depot_set for n in r)
    ]
    # Ensure now that every removed node is not in any route
    for n in removed_nodes:
        for route in destroyed_solution:
            assert n not in route
    # Ensure that remaining customers are correct
    customers_in_sol = set()
    for route in destroyed_solution:
        customers_in_sol.update([n for n in route if n not in depot_set])
    check_union = customers_in_sol | removed_nodes
    assert customers_in_sol.isdisjoint(removed_nodes)
    assert set(all_customers) == check_union
    assert len(removed_nodes) == n_remove
    destroyed_solution = [
        r for r in destroyed_solution if any(n not in depot_set for n in r)
    ]
    return list(removed_nodes), destroyed_solution

def insert(destroyed_solution, removed_nodes, instance, dist_matrix):
    depot_set = set(instance['depot'])
    demands = instance['demands']
    capacity = instance['capacity'][0]
    distance_limit = instance['distance_limit'][0]
    n_customers = len([d for d in demands if d > 0])
    routes = [route[:] for route in destroyed_solution]
    def route_demand(route):
        return sum(demands[n - 1] for n in route if n not in depot_set)
    def route_distance(route):
        c = 0.0
        for i in range(len(route) - 1):
            c += dist_matrix[route[i] - 1][route[i + 1] - 1]
        return c
    def is_feasible(route):
        if len(route) < 2:
            return False
        load = route_demand(route)
        if load - 1e-6 > capacity:
            return False
        d = route_distance(route)
        if d - 1e-6 > distance_limit:
            return False
        return True
    # Track all customers in current solution at start (for uniqueness check later)
    all_customers = set([i + 1 for i, d in enumerate(demands) if d > 0])
    # Insert each removed node exactly once
    for node in removed_nodes:
        best_cost_increase = float('inf')
        best_insert = None
        best_new_route = None
        # Try inserting into existing routes
        for ridx, route in enumerate(routes):
            # Only consider real (non-empty) routes
            for pos in range(1, len(route)):
                if node in route:
                    continue
                new_route = route[:pos] + [node] + route[pos:]
                if not is_feasible(new_route):
                    continue
                old_cost = route_distance(route)
                new_cost = route_distance(new_route)
                cost_increase = new_cost - old_cost
                if cost_increase < best_cost_increase - 1e-9:
                    best_cost_increase = cost_increase
                    best_insert = (ridx, pos)
                    best_new_route = new_route
        # Try as a new route
        depot = instance['depot'][0]
        new_route_try = [depot, node, depot]
        if is_feasible(new_route_try) and best_new_route is None:
            routes.append(new_route_try)
        elif best_new_route is not None:
            ridx, pos = best_insert
            routes[ridx] = routes[ridx][:pos] + [node] + routes[ridx][pos:]
        else:
            # If neither is feasible, this means it's impossible (should not happen)
            raise RuntimeError("Unable to reinsert node %d feasibly into any route or as a new route." % node)
    # Clean up: remove duplicate customers (shouldn't exist) and empty routes, and redundant depots
    seen = set()
    clean_routes = []
    for r in routes:
        cleaned = []
        prev = None
        for n in r:
            if n in depot_set:
                if prev != n:
                    cleaned.append(n)
            else:
                if n not in seen:
                    cleaned.append(n)
                    seen.add(n)
            prev = n
        # Only keep non-empty customer route
        this_customers = [n for n in cleaned if n not in depot_set]
        if len(this_customers) > 0:
            clean_routes.append(cleaned)
    # Final customer uniqueness enforcement
    customers_in_solution = []
    for route in clean_routes:
        customers_in_solution += [n for n in route if n not in depot_set]
    # Remove extra customers if any duplicates from accidental insertion, enforce true set
    unique_customers = set()
    for idx, route in enumerate(clean_routes):
        route_new = [route[0]]
        for n in route[1:-1]:
            if n not in depot_set and n in unique_customers:
                continue
            if n not in depot_set:
                unique_customers.add(n)
            route_new.append(n)
        route_new.append(route[-1])
        clean_routes[idx] = route_new
    # After cleaning, check each customer is present exactly once (if not, remove any extra occurrence)
    final_customers = []
    for r in clean_routes:
        final_customers += [n for n in r if n not in depot_set]
    missing = all_customers - set(final_customers)
    assert len(missing) == 0, f"Missing customers after insert: {missing}"
    extra = set([n for n in final_customers if final_customers.count(n) > 1])
    assert len(extra) == 0, f"Duplicated customers after insert: {extra}"
    # Remove routes which are depot only or no customers
    final_routes = []
    for r in clean_routes:
        customers = [n for n in r if n not in depot_set]
        if len(customers) > 0:
            final_routes.append(r)
    return final_routes

def validate(solution, instance, dist_matrix):
    depot_set = set(instance['depot'])
    demands = instance['demands']
    capacity = instance['capacity'][0]
    distance_limit = instance['distance_limit'][0]
    n_nodes = len(demands)
    depot_id = instance['depot'][0]
    for idx, route in enumerate(solution):
        if len(route) < 2:
            print(f"Depot constraint violated: Route {idx} has less than 2 nodes.")
            return False
        if route[0] != depot_id or route[-1] != depot_id:
            print(f"Depot constraint violated: Route {idx} does not start/end at the depot.")
            return False
    for idx, route in enumerate(solution):
        load = 0.0
        for n in route:
            if n not in depot_set:
                if n < 1 or n > n_nodes:
                    print(f"Route {idx} contains an invalid node id {n}.")
                    return False
                load += demands[n - 1]
        if load - 1e-6 > capacity:
            print(f"Capacity constraint violated: Route {idx} load {load} exceeds capacity {capacity}.")
            return False
    for idx, route in enumerate(solution):
        dist = 0.0
        for i in range(len(route) - 1):
            from_n = route[i]
            to_n = route[i + 1]
            if not (1 <= from_n <= n_nodes) or not (1 <= to_n <= n_nodes):
                print(f"Route {idx} contains invalid node ids {from_n}->{to_n}.")
                return False
            dist += dist_matrix[from_n - 1][to_n - 1]
        if dist - 1e-6 > distance_limit:
            print(f"Distance constraint violated: Route {idx} distance {dist} exceeds limit {distance_limit}.")
            return False
    visits = [0] * n_nodes
    for idx, route in enumerate(solution):
        for n in route:
            if n in depot_set:
                continue
            if not (1 <= n <= n_nodes):
                print(f"Route {idx} contains invalid node id {n} during visit check.")
                return False
            visits[n - 1] += 1
    violated = False
    for i in range(n_nodes):
        if demands[i] > 0:
            if visits[i] != 1:
                print(f"Visit constraint violated: Node {i + 1} (demand {demands[i]}) visited {visits[i]} times.")
                violated = True
        else:
            if visits[i] != 0:
                print(f"Visit constraint violated: Depot/non-customer node {i + 1} (demand 0) visited {visits[i]} times.")
                violated = True
    if violated:
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
        raise Exception("Initial solution is not feasible.")
    current_cost = cost(route=current_solution, dist_matrix=dist_matrix)
    best_solution = current_solution
    best_cost = current_cost
    print(f"the initial process is successful, the initial cost is {best_cost}")

    for step in range(iteration):
        ratio = random.uniform(0, 0.2)
        removed_nodes, destroyed_solution = destroy(instance, dist_matrix, solution=current_solution, ratio=ratio)
        current_solution = insert(destroyed_solution, removed_nodes, instance, dist_matrix)
        if not validate(solution=current_solution, instance=instance, dist_matrix=dist_matrix):
            raise Exception(f"Solution at step {step} is not feasible.")
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