import math
import random
import argparse
from typing import List, Dict, Any
from copy import deepcopy

def read_vrp(path: str) -> Dict[str, List[Any]]:
    depot: List[int] = []
    node_coordinates: List[List[float]] = []
    demands: List[float] = []
    vehicle_capacity: List[float] = []
    distance_limit: List[float] = []
    dimension: int = None

    node_coordinates_dict: Dict[int, List[float]] = {}
    demands_dict: Dict[int, float] = {}

    with open(path, "r") as f:
        lines = [l.rstrip("\n") for l in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        lkey = line.split(":")[0].upper() if ":" in line else line.upper()
        if lkey.startswith("DIMENSION"):
            if ":" in line:
                dimension = int(line.split(":")[1].strip())
            else:
                dimension = int(line.split()[-1])
            i += 1
        elif lkey.startswith("CAPACITY"):
            if ":" in line:
                vc = float(line.split(":")[1].strip())
            else:
                vc = float(line.split()[-1])
            vehicle_capacity = [vc]
            i += 1
        elif lkey.startswith("DISTANCE_LIMIT"):
            if ":" in line:
                dl = float(line.split(":")[1].strip())
            else:
                dl = float(line.split()[-1])
            distance_limit = [dl]
            i += 1
        elif lkey.startswith("NODE_COORD_SECTION"):
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if not l or l.upper().startswith("DEMAND_SECTION") or l.upper().startswith("EOF"):
                    break
                tokens = l.split()
                if len(tokens) == 3:
                    idx = int(tokens[0])
                    x, y = float(tokens[1]), float(tokens[2])
                    node_coordinates_dict[idx] = [x, y]
                i += 1
        elif lkey.startswith("DEMAND_SECTION"):
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if not l or l.upper().startswith("DEPOT_SECTION") or l.upper().startswith("EOF"):
                    break
                tokens = l.split()
                if len(tokens) == 2:
                    idx = int(tokens[0])
                    d = float(tokens[1])
                    demands_dict[idx] = d
                i += 1
        elif lkey.startswith("DEPOT_SECTION"):
            i += 1
            depots = []
            while i < len(lines):
                l = lines[i].strip()
                if l == "-1" or l.upper() == "EOF":
                    break
                if l:
                    depots.append(int(l))
                i += 1
            depot = depots
            i += 1
        else:
            i += 1

    required_keys = ["depot", "node_coordinates", "demands", "vehicle_capacity", "distance_limit"]
    if dimension is None or not vehicle_capacity or not distance_limit or not depot:
        raise ValueError("Missing required fields in .vrp file.")

    if depot[0] != 1:
        raise ValueError("Depot must be node 1 for strict one-based ID convention")

    node_coordinates = [None] * dimension
    demands = [None] * dimension
    for idx in range(1, dimension + 1):
        if idx not in node_coordinates_dict or idx not in demands_dict:
            raise ValueError(f"Missing coordinate or demand for node {idx}.")
        node_coordinates[idx - 1] = node_coordinates_dict[idx]
        demands[idx - 1] = demands_dict[idx]

    output = {
        "depot": depot,
        "node_coordinates": node_coordinates,
        "demands": demands,
        "vehicle_capacity": vehicle_capacity,
        "distance_limit": distance_limit
    }
    if set(output.keys()) != set(required_keys):
        raise ValueError(f"Unexpected fields in output dictionary: {set(output.keys()) ^ set(required_keys)}")
    for key in required_keys:
        if key not in output:
            raise ValueError(f"Field {key} missing.")
        if not isinstance(output[key], list):
            raise ValueError(f"Field {key} not a list.")
        if key == "depot":
            if output[key][0] != 1:
                raise ValueError("Depot must be node 1.")
    return output

def distance(coords: List[List[float]]) -> List[List[float]]:
    n = len(coords)
    dist_matrix = []
    for i in range(n):
        x1, y1 = coords[i]
        row = []
        for j in range(n):
            x2, y2 = coords[j]
            row.append(math.hypot(x2 - x1, y2 - y1))
        dist_matrix.append(row)
    return dist_matrix

def cost(route, dist_matrix):
    if isinstance(route, list) and route and isinstance(route[0], list):
        # route is a list of routes
        return sum(cost(r, dist_matrix) for r in route)
    # Single open route: sum depot to first, then first to 2nd, ..., no return to depot
    if not route or len(route) < 2:
        return 0.0
    total = 0.0
    for i in range(len(route) - 1):
        n1, n2 = route[i], route[i + 1]
        total += dist_matrix[n1 - 1][n2 - 1]
    return total

def initial(instance: Dict[str, List[Any]], dist_matrix: List[List[float]]) -> List[List[int]]:
    depot_list = instance["depot"]
    if not depot_list:
        raise ValueError("Depot list is empty.")
    depot = depot_list[0]
    demands = instance["demands"]
    cap = instance["vehicle_capacity"][0]
    L = instance["distance_limit"][0]
    num_nodes = len(instance["node_coordinates"])
    all_nodes = list(range(1, num_nodes + 1))
    customer_nodes = [nid for nid in all_nodes if nid not in depot_list]
    unassigned = set(customer_nodes)
    routes: List[List[int]] = []
    depot_set = set(depot_list)

    while unassigned:
        route = [depot]
        route_load = 0.0
        route_length = 0.0
        last_node = depot
        available_nodes = set(unassigned)
        while True:
            best_next = None
            best_cost = float('inf')
            # Find the nearest feasible customer
            for nid in available_nodes:
                demand_add = demands[nid - 1]
                tmp_load = route_load + demand_add
                tmp_len = route_length + dist_matrix[last_node - 1][nid - 1]
                if tmp_load > cap + 1e-9:
                    continue
                if tmp_len > L + 1e-9:
                    continue
                dcost = dist_matrix[last_node - 1][nid - 1]
                if dcost < best_cost:
                    best_cost = dcost
                    best_next = nid

            # OCVRPL: After each assignment, check if it's better to start a new route
            # Compute the minimal depot-to-unassigned-customer distance
            min_depot_to_cust = float('inf')
            if available_nodes:
                for nid in available_nodes:
                    dcost = dist_matrix[depot - 1][nid - 1]
                    if dcost < min_depot_to_cust:
                        min_depot_to_cust = dcost

            # If there is a next possible customer, check open split criterion
            # Only build out route if connecting from current last node is "more attractive" than opening a new route from depot;
            # if depot->some-unassigned is closer than last_node->best_next, then do NOT continue, open new route instead.
            if (
                best_next is None or
                (len(route) > 1 and min_depot_to_cust + 1e-9 < best_cost - 1e-9)
            ):
                break
            # Otherwise, add customer to route
            route.append(best_next)
            route_load += demands[best_next - 1]
            route_length += dist_matrix[last_node - 1][best_next - 1]
            last_node = best_next
            unassigned.remove(best_next)
            available_nodes = set(unassigned)
        if len(route) > 1:
            routes.append(route)
        else:
            # Can't build valid route, try to allocate single customer to new route if feasible
            found_singleton = False
            for singleton_nid in sorted(unassigned):
                demand = demands[singleton_nid - 1]
                dist_from_depot = dist_matrix[depot - 1][singleton_nid - 1]
                if demand <= cap + 1e-9 and dist_from_depot <= L + 1e-9:
                    routes.append([depot, singleton_nid])
                    unassigned.remove(singleton_nid)
                    found_singleton = True
                    break
            if not found_singleton:
                raise RuntimeError(f"No feasible route could be constructed for remaining customers {unassigned}; check instance/constraints.")

    assigned_customers = set()
    for r in routes:
        if r[0] != depot:
            raise AssertionError("Route does not start at depot")
        if any(n in depot_set for n in r[1:]):
            raise AssertionError("Depot node found after start in route")
        route_load = sum(demands[n - 1] for n in r[1:])
        route_len = cost(r, dist_matrix)
        if not route_load <= cap + 1e-6:
            raise AssertionError("Capacity exceeded in route")
        if not route_len <= L + 1e-6:
            raise AssertionError("Distance limit exceeded in route")
        assigned_customers.update(r[1:])
        if r.count(depot) != 1 or any(nn == depot for nn in r[1:]):
            raise AssertionError("Depot node position wrong in route")
    if assigned_customers != set(customer_nodes):
        raise AssertionError("Not all customers are assigned")
    return routes

def destroy(instance: Dict[str, List[Any]], dist_matrix: List[List[float]], solution: List[List[int]], ratio: float):
    depot_list = instance["depot"]
    depot = depot_list[0]
    n_nodes = len(instance["node_coordinates"])
    customer_nodes = [nid for nid in range(1, n_nodes + 1) if nid not in depot_list]
    num_remove = int(round(len(customer_nodes) * ratio))
    if num_remove < 1:
        num_remove = 1

    destroyed_solution = deepcopy(solution)

    assigned_customers = []
    route_map = {}
    for ridx, route in enumerate(destroyed_solution):
        for p, node in enumerate(route[1:], 1):
            if node == depot:
                continue
            assigned_customers.append(node)
            route_map[node] = (ridx, p)

    if not assigned_customers:
        return [], deepcopy(solution)

    center = random.choice(assigned_customers)

    customer_dists = []
    for nid in customer_nodes:
        if nid == center:
            dist = 0
        else:
            dist = dist_matrix[center - 1][nid - 1]
        customer_dists.append((dist, nid))
    customer_dists.sort()
    ordered_neighbors = [nid for _, nid in customer_dists if nid != depot]

    removed_nodes = []
    destroyed_routes = set()
    positions_to_remove = set()

    for neighbor in [center] + ordered_neighbors:
        if len(removed_nodes) >= num_remove:
            break
        if neighbor not in route_map:
            continue
        ridx, pos = route_map[neighbor]
        if ridx in destroyed_routes:
            continue
        route = destroyed_solution[ridx]
        route_length = len(route)
        max_block = min(num_remove - len(removed_nodes), route_length - 1)
        if max_block < 1:
            continue
        block_len = random.randint(1, max_block)
        left_range = max(1, pos - block_len + 1)
        right_range = min(pos, route_length - block_len)
        if right_range < left_range:
            start = left_range
        else:
            start = random.randint(left_range, right_range)
        end = start + block_len
        for pblock in range(start, end):
            if (ridx, pblock) in positions_to_remove:
                continue
            removed_nodes.append(route[pblock])
            positions_to_remove.add((ridx, pblock))
            if len(removed_nodes) >= num_remove:
                break
        destroyed_routes.add(ridx)

    for route_idx in range(len(destroyed_solution)):
        route = destroyed_solution[route_idx]
        new_route = [route[0]] + [n for i, n in enumerate(route[1:], start=1) if (route_idx, i) not in positions_to_remove]
        destroyed_solution[route_idx] = new_route
    destroyed_solution = [r for r in destroyed_solution if len(r) > 1]

    removed_set = set(removed_nodes)
    assigned_now = set(n for r in destroyed_solution for n in r[1:])
    all_cust = set(customer_nodes)
    assert removed_set.issubset(all_cust)
    assert assigned_now.issubset(all_cust)
    assert not (removed_set & assigned_now)
    assert (removed_set | assigned_now) == all_cust
    assert len(removed_set) == len(removed_nodes)
    assert depot not in removed_set
    return removed_nodes, destroyed_solution

def insert(destroyed_solution: List[List[int]], removed_nodes: List[int], instance: Dict[str, List[Any]], dist_matrix: List[List[float]]) -> List[List[int]]:
    solution = deepcopy(destroyed_solution)
    depot_list = instance["depot"]
    depot = depot_list[0]
    demands = instance["demands"]
    cap = instance["vehicle_capacity"][0]
    L = instance["distance_limit"][0]
    node_n = len(instance["node_coordinates"])
    all_nodes = list(range(1, node_n + 1))
    depot_set = set(depot_list)
    customer_nodes = [nid for nid in all_nodes if nid not in depot_set]

    for node in removed_nodes:
        if node in depot_set:
            raise AssertionError("Depot cannot be reinserted")
        demand_node = demands[node - 1]
        is_new_route_feasible = demand_node <= cap + 1e-9 and dist_matrix[depot - 1][node - 1] <= L + 1e-9
        best_route = None
        best_pos = None
        best_delta = float("inf")
        for ridx, route in enumerate(solution):
            if depot in route[1:]:
                continue
            route_load = sum(demands[n - 1] for n in route[1:])
            if route_load + demand_node > cap + 1e-9:
                continue
            for pos in range(1, len(route) + 1):
                new_route = route[:pos] + [node] + route[pos:]
                if new_route[0] != depot or depot in new_route[1:]:
                    continue
                new_load = sum(demands[n - 1] for n in new_route[1:])
                if new_load > cap + 1e-9:
                    continue
                route_len = cost(new_route, dist_matrix)
                if route_len > L + 1e-9:
                    continue
                orig_len = cost(route, dist_matrix)
                delta = route_len - orig_len
                # Open VRP: ensure open-termination logic
                if pos == 1 and len(route) == 1:
                    pass
                if delta < best_delta:
                    best_route = ridx
                    best_pos = pos
                    best_delta = delta
        if is_new_route_feasible and (best_route is None or dist_matrix[depot - 1][node - 1] + 1e-9 < best_delta - 1e-9):
            solution.append([depot, node])
            continue
        if best_route is not None and best_pos is not None:
            solution[best_route] = solution[best_route][:best_pos] + [node] + solution[best_route][best_pos:]
        else:
            raise RuntimeError(f"No feasible way to insert node {node} in any route or as new.")

    assigned = set()
    for route in solution:
        assert route[0] == depot, "Route does not start at depot"
        assert depot not in route[1:], "Depot appears at non-start position"
        total_dem = sum(demands[n - 1] for n in route[1:])
        assert total_dem <= cap + 1e-6, "Capacity violated"
        route_len = cost(route, dist_matrix)
        assert route_len <= L + 1e-6, "Distance limit violated"
        for n in route[1:]:
            assert n not in depot_set, "Depot assigned as customer"
            assigned.add(n)
        assert route.count(depot) == 1, "Depot appears more than once"
    assert assigned == set(customer_nodes)
    return solution

def validation(solution, instance, dist_matrix):
    depot_list = instance["depot"]
    if not depot_list:
        print("Depot not defined in instance.")
        return False
    depot = depot_list[0]
    if depot != 1:
        print("Depot id must be 1.")
        return False
    demands = instance["demands"]
    cap = instance["vehicle_capacity"][0]
    L = instance["distance_limit"][0]
    node_n = len(instance["node_coordinates"])
    all_nodes = set(range(1, node_n + 1))
    depot_set = set(depot_list)
    customer_nodes = all_nodes - depot_set

    if not isinstance(solution, list) or not all(isinstance(r, list) for r in solution):
        print("Solution is not a list of routes.")
        return False
    for route in solution:
        if not route or not all(isinstance(n, int) for n in route):
            print("Route is not a list of node IDs.")
            return False
    for idx, route in enumerate(solution):
        if len(route) < 2:
            print(f"Route {idx} is too short or only contains the depot.")
            return False
        if route[0] != depot:
            print(f"Route {idx} does not start at the depot {depot}.")
            return False
        if depot in route[1:]:
            print(f"Route {idx} contains depot at a non-starting position.")
            return False
        if route.count(depot) != 1:
            print(f"Depot appears more than once in route {idx}.")
            return False

    assigned_customers = []
    for r in solution:
        for n in r[1:]:
            if n in depot_set:
                print(f"Customer node {n} in route appears in depot set.")
                return False
            if n < 1 or n > node_n:
                print(f"Invalid node {n} in route.")
                return False
            assigned_customers.append(n)
    assigned_set = set(assigned_customers)
    if assigned_set != customer_nodes:
        missing = sorted(list(customer_nodes - assigned_set))
        extra = sorted(list(assigned_set - customer_nodes))
        if missing:
            print(f"Missing customers from solution: {missing}")
        if extra:
            print(f"Extra/non-customer nodes assigned: {extra}")
        return False
    if len(assigned_customers) != len(assigned_set):
        seen, dups = set(), set()
        for n in assigned_customers:
            if n in seen:
                dups.add(n)
            seen.add(n)
        print(f"Customers assigned more than once: {sorted(list(dups))}")
        return False

    for idx, route in enumerate(solution):
        load = sum(demands[n - 1] for n in route[1:])
        if load > cap + 1e-6:
            print(f"Route {idx} exceeds capacity: load {load} > {cap}")
            return False

    for idx, route in enumerate(solution):
        route_len = cost(route, dist_matrix)
        if route_len > L + 1e-6:
            print(f"Route {idx} exceeds distance limit: {route_len} > {L}")
            return False

    for idx, route in enumerate(solution):
        for i in range(1, len(route)):
            if route[i] == depot:
                print(f"Depot appears at non-start position in route {idx}.")
                return False
        for n in route:
            if n < 1 or n > node_n:
                print(f"Node {n} in route {idx} is out of bounds.")
                return False
    return True

validate = validation

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
    best_solution = deepcopy(current_solution)
    best_cost = current_cost

    for step in range(iteration):
        ratio = random.uniform(0.0001, 0.1999)
        removed_nodes, destroyed_solution = destroy(instance, dist_matrix, solution=current_solution, ratio=ratio)
        current_solution = insert(destroyed_solution, removed_nodes, instance, dist_matrix)
        if not validate(solution=current_solution, instance=instance, dist_matrix=dist_matrix):
            raise Exception("Intermediate solution in iteration is not valid!")
        current_cost = cost(route=current_solution, dist_matrix=dist_matrix)
        if current_cost <= best_cost:
            best_solution = deepcopy(current_solution)
            best_cost = current_cost
        else:
            p = random.uniform(0, 1)
            threshold = math.exp(-(current_cost - best_cost) * iteration * 10 / (iteration - step + 1))
            if p > threshold:
                current_solution = deepcopy(best_solution)
                current_cost = best_cost

    print(f"the process is successful, the best cost is {best_cost}")