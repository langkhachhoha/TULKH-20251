"""
Improved Local Search Algorithm for VRPTW
Mode 1: 2-opt with restart and first improvement
Mode 2: Variable neighborhood descent (2-opt + insert + swap)
Mode 3: Iterated local search with perturbation
Mode 4: Simulated annealing with 2-opt
Mode 5: GRASP (Greedy Randomized Adaptive Search)
Mode 6: Late acceptance hill climbing
"""

import random
import sys
import copy
import math

MODE = 1  # Change this to 1, 2, 3, 4, 5, or 6

def read_input():
    n = int(input())
    
    customers = []
    for i in range(n):
        e, l, d = map(int, input().split())
        customers.append({'id': i+1, 'earliest': e, 'latest': l, 'duration': d})
    
    travel_time = []
    for i in range(n + 1):
        row = list(map(int, input().split()))
        travel_time.append(row)
    
    return n, customers, travel_time

def calculate_total_time(route, customers, travel_time, t0=0):
    """Calculate total travel time and check feasibility"""
    current_time = t0
    total_travel = 0
    prev = 0
    
    for idx in route:
        travel = travel_time[prev][idx]
        arrival_time = current_time + travel
        
        customer = customers[idx - 1]
        
        # Check if we can arrive before latest time
        if arrival_time > customer['latest']:
            return float('inf')  # Infeasible
        
        # Wait if we arrive too early
        start_service = max(arrival_time, customer['earliest'])
        
        # Complete service
        current_time = start_service + customer['duration']
        total_travel += travel
        prev = idx
    
    return total_travel

def is_feasible(route, customers, travel_time, t0=0):
    """Check if route is feasible"""
    return calculate_total_time(route, customers, travel_time, t0) != float('inf')

def generate_initial_solution(n, customers, travel_time, t0=0, strategy='nearest'):
    """Generate initial feasible solution with different strategies"""
    
    if strategy == 'nearest':
        # Nearest neighbor
        route = []
        visited = set()
        current_time = t0
        current_pos = 0
        
        while len(route) < n:
            best_next = -1
            best_cost = float('inf')
            
            for i in range(1, n + 1):
                if i in visited:
                    continue
                
                travel = travel_time[current_pos][i]
                arrival_time = current_time + travel
                customer = customers[i - 1]
                
                if arrival_time <= customer['latest']:
                    if travel < best_cost:
                        best_cost = travel
                        best_next = i
            
            if best_next == -1:
                for i in range(1, n + 1):
                    if i not in visited:
                        route.append(i)
                        visited.add(i)
                break
            
            route.append(best_next)
            visited.add(best_next)
            
            customer = customers[best_next - 1]
            arrival_time = current_time + best_cost
            start_service = max(arrival_time, customer['earliest'])
            current_time = start_service + customer['duration']
            current_pos = best_next
    
    elif strategy == 'earliest':
        # Sort by earliest time window
        route = sorted(range(1, n + 1), key=lambda x: customers[x-1]['earliest'])
    
    elif strategy == 'latest':
        # Sort by latest time window
        route = sorted(range(1, n + 1), key=lambda x: customers[x-1]['latest'])
    
    else:
        # Random
        route = list(range(1, n + 1))
        random.shuffle(route)
    
    return route

def two_opt_swap(route, i, j):
    """Swap positions i and j"""
    new_route = route[:]
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def three_opt_swap(route, i, j, k):
    """Rotate three positions"""
    new_route = route[:]
    new_route[i], new_route[j], new_route[k] = new_route[k], new_route[i], new_route[j]
    return new_route

def insert_move(route, i, j):
    """Remove element at i and insert at j"""
    new_route = route[:]
    element = new_route.pop(i)
    new_route.insert(j, element)
    return new_route

def reverse_segment(route, i, j):
    """Reverse segment from i to j"""
    new_route = route[:]
    if i > j:
        i, j = j, i
    new_route[i:j+1] = reversed(new_route[i:j+1])
    return new_route

def random_swap(route):
    """Random swap of two positions"""
    new_route = route[:]
    i, j = random.sample(range(len(route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def local_search_mode1(route, customers, travel_time, max_iter=10000):
    """Mode 1: 2-opt with restart and first improvement"""
    best_route = route[:]
    best_cost = calculate_total_time(best_route, customers, travel_time)
    
    n = len(route)
    restarts = 0
    max_restarts = 5
    no_improve_count = 0
    
    while restarts < max_restarts:
        improved = True
        iterations = 0
        current_route = best_route[:]
        current_cost = best_cost
        
        # Local search with first improvement
        while improved and iterations < max_iter // max_restarts:
            improved = False
            iterations += 1
            
            # Random order to avoid bias
            indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(indices)
            
            for i, j in indices[:min(len(indices), n * 5)]:  # Limit neighborhood size
                new_route = two_opt_swap(current_route, i, j)
                new_cost = calculate_total_time(new_route, customers, travel_time)
                
                if new_cost < current_cost:
                    current_route = new_route
                    current_cost = new_cost
                    improved = True
                    
                    if current_cost < best_cost:
                        best_route = current_route[:]
                        best_cost = current_cost
                        no_improve_count = 0
                    break
        
        # Restart with perturbation
        restarts += 1
        no_improve_count += 1
        
        if no_improve_count >= 2 and restarts < max_restarts:
            # Perturb best solution
            current_route = best_route[:]
            for _ in range(random.randint(2, 5)):
                i, j = random.sample(range(n), 2)
                current_route[i], current_route[j] = current_route[j], current_route[i]
    
    return best_route, best_cost

def local_search_mode2(route, customers, travel_time, max_iter=10000):
    """Mode 2: Variable neighborhood descent (2-opt + insert + swap)"""
    best_route = route[:]
    best_cost = calculate_total_time(best_route, customers, travel_time)
    
    n = len(route)
    neighborhoods = ['swap', 'insert', 'reverse']
    
    iterations = 0
    while iterations < max_iter:
        improved = False
        iterations += 1
        
        # Try each neighborhood in sequence
        for neighborhood in neighborhoods:
            if neighborhood == 'swap':
                # 2-opt swap
                indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
                random.shuffle(indices)
                
                for i, j in indices[:min(len(indices), n * 3)]:
                    new_route = two_opt_swap(best_route, i, j)
                    new_cost = calculate_total_time(new_route, customers, travel_time)
                    
                    if new_cost < best_cost:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
                        break
            
            elif neighborhood == 'insert':
                # Insert move
                indices = [(i, j) for i in range(n) for j in range(n) if i != j]
                random.shuffle(indices)
                
                for i, j in indices[:min(len(indices), n * 3)]:
                    new_route = insert_move(best_route, i, j)
                    new_cost = calculate_total_time(new_route, customers, travel_time)
                    
                    if new_cost < best_cost:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
                        break
            
            else:
                # Reverse segment
                indices = [(i, j) for i in range(n) for j in range(i + 2, n)]
                random.shuffle(indices)
                
                for i, j in indices[:min(len(indices), n * 3)]:
                    new_route = reverse_segment(best_route, i, j)
                    new_cost = calculate_total_time(new_route, customers, travel_time)
                    
                    if new_cost < best_cost:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_route, best_cost

def perturbation(route, strength=3):
    """Perturb a solution"""
    new_route = route[:]
    n = len(route)
    
    for _ in range(strength):
        move_type = random.choice(['swap', 'insert', 'reverse'])
        
        if move_type == 'swap':
            i, j = random.sample(range(n), 2)
            new_route[i], new_route[j] = new_route[j], new_route[i]
        elif move_type == 'insert':
            i, j = random.sample(range(n), 2)
            element = new_route.pop(i)
            new_route.insert(j, element)
        else:
            i, j = sorted(random.sample(range(n), 2))
            new_route[i:j+1] = reversed(new_route[i:j+1])
    
    return new_route

def local_search_mode3(route, customers, travel_time, max_iter=10000):
    """Mode 3: Iterated local search with perturbation"""
    best_route = route[:]
    best_cost = calculate_total_time(best_route, customers, travel_time)
    
    n = len(route)
    current_route = best_route[:]
    
    num_iterations = 0
    no_improve = 0
    
    while num_iterations < max_iter:
        # Local search phase
        improved = True
        local_iter = 0
        
        while improved and local_iter < 100:
            improved = False
            local_iter += 1
            
            # Try 2-opt moves
            indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(indices)
            
            for i, j in indices[:min(len(indices), n * 4)]:
                new_route = two_opt_swap(current_route, i, j)
                new_cost = calculate_total_time(new_route, customers, travel_time)
                
                current_cost = calculate_total_time(current_route, customers, travel_time)
                if new_cost < current_cost:
                    current_route = new_route
                    improved = True
                    
                    if new_cost < best_cost:
                        best_route = new_route[:]
                        best_cost = new_cost
                        no_improve = 0
                    break
        
        num_iterations += local_iter
        no_improve += 1
        
        # Perturbation phase
        if no_improve >= 5:
            # Strong perturbation
            current_route = perturbation(best_route, strength=5)
            no_improve = 0
        else:
            # Weak perturbation
            current_route = perturbation(current_route, strength=2)
    
    return best_route, best_cost

def local_search_mode4(route, customers, travel_time, max_iter=10000):
    """Mode 4: Simulated annealing with 2-opt"""
    best_route = route[:]
    best_cost = calculate_total_time(best_route, customers, travel_time)
    
    current_route = best_route[:]
    current_cost = best_cost
    
    n = len(route)
    
    # SA parameters
    temp = 100.0
    cooling_rate = 0.995
    min_temp = 0.1
    
    iterations = 0
    
    while temp > min_temp and iterations < max_iter:
        iterations += 1
        
        # Generate neighbor
        move_type = random.choice(['swap', 'insert', 'reverse'])
        
        if move_type == 'swap':
            i, j = random.sample(range(n), 2)
            neighbor = two_opt_swap(current_route, i, j)
        elif move_type == 'insert':
            i, j = random.sample(range(n), 2)
            neighbor = insert_move(current_route, i, j)
        else:
            i, j = sorted(random.sample(range(n), 2))
            neighbor = reverse_segment(current_route, i, j)
        
        neighbor_cost = calculate_total_time(neighbor, customers, travel_time)
        
        # Accept or reject
        delta = neighbor_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_route = neighbor
            current_cost = neighbor_cost
            
            if current_cost < best_cost:
                best_route = current_route[:]
                best_cost = current_cost
        
        # Cool down
        temp *= cooling_rate
    
    return best_route, best_cost

def greedy_randomized_construction(n, customers, travel_time, alpha=0.3):
    """GRASP construction phase with randomization"""
    route = []
    visited = set()
    current_time = 0
    current_pos = 0
    
    while len(route) < n:
        candidates = []
        
        for i in range(1, n + 1):
            if i in visited:
                continue
            
            travel = travel_time[current_pos][i]
            arrival_time = current_time + travel
            customer = customers[i - 1]
            
            if arrival_time <= customer['latest']:
                candidates.append((i, travel))
        
        if not candidates:
            # Add remaining
            for i in range(1, n + 1):
                if i not in visited:
                    route.append(i)
                    visited.add(i)
            break
        
        # Sort by cost
        candidates.sort(key=lambda x: x[1])
        
        # Restricted candidate list (RCL)
        min_cost = candidates[0][1]
        max_cost = candidates[-1][1]
        threshold = min_cost + alpha * (max_cost - min_cost)
        
        rcl = [c for c in candidates if c[1] <= threshold]
        
        # Random selection from RCL
        chosen, cost = random.choice(rcl)
        
        route.append(chosen)
        visited.add(chosen)
        
        customer = customers[chosen - 1]
        arrival_time = current_time + cost
        start_service = max(arrival_time, customer['earliest'])
        current_time = start_service + customer['duration']
        current_pos = chosen
    
    return route

def local_search_mode5(route, customers, travel_time, max_iter=10000):
    """Mode 5: GRASP (Greedy Randomized Adaptive Search)"""
    best_route = None
    best_cost = float('inf')
    
    n = len(route)
    num_constructions = max(10, max_iter // 1000)
    
    for _ in range(num_constructions):
        # Construction phase
        current_route = greedy_randomized_construction(n, customers, travel_time, alpha=0.3)
        
        # Local search phase (2-opt)
        improved = True
        local_iter = 0
        max_local_iter = max_iter // num_constructions
        
        while improved and local_iter < max_local_iter:
            improved = False
            local_iter += 1
            
            indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(indices)
            
            for i, j in indices[:min(len(indices), n * 4)]:
                new_route = two_opt_swap(current_route, i, j)
                new_cost = calculate_total_time(new_route, customers, travel_time)
                current_cost = calculate_total_time(current_route, customers, travel_time)
                
                if new_cost < current_cost:
                    current_route = new_route
                    improved = True
                    break
        
        current_cost = calculate_total_time(current_route, customers, travel_time)
        if current_cost < best_cost:
            best_route = current_route[:]
            best_cost = current_cost
    
    return best_route, best_cost

def local_search_mode6(route, customers, travel_time, max_iter=10000):
    """Mode 6: Late acceptance hill climbing"""
    best_route = route[:]
    best_cost = calculate_total_time(best_route, customers, travel_time)
    current_route = route[:]
    
    n = len(route)
    
    # Late acceptance list - stores costs from previous iterations
    L = 100  # List length
    acceptance_list = [best_cost] * L
    list_index = 0
    
    for iteration in range(max_iter):
        # Generate neighbor
        move_type = random.choice(['swap', 'insert', 'reverse'])
        
        if move_type == 'swap':
            i, j = random.sample(range(n), 2)
            neighbor = two_opt_swap(current_route, i, j)
        elif move_type == 'insert':
            i, j = random.sample(range(n), 2)
            neighbor = insert_move(current_route, i, j)
        else:
            i, j = sorted(random.sample(range(n), 2))
            if i == j:
                continue
            neighbor = reverse_segment(current_route, i, j)
        
        neighbor_cost = calculate_total_time(neighbor, customers, travel_time)
        current_cost = calculate_total_time(current_route, customers, travel_time)
        
        # Late acceptance criterion: accept if better than cost L iterations ago
        if neighbor_cost <= acceptance_list[list_index] or neighbor_cost <= current_cost:
            current_route = neighbor
            current_cost = neighbor_cost
            
            if current_cost < best_cost:
                best_route = current_route[:]
                best_cost = current_cost
        
        # Update acceptance list
        acceptance_list[list_index] = current_cost
        list_index = (list_index + 1) % L
    
    return best_route, best_cost

def solve(mode):
    n, customers, travel_time = read_input()
    
    # Try multiple initial solutions and pick the best
    best_overall_route = None
    best_overall_cost = float('inf')
    
    strategies = ['nearest', 'earliest', 'latest']
    
    for strategy in strategies:
        initial_route = generate_initial_solution(n, customers, travel_time, strategy=strategy)
        
        # Apply local search based on mode
        if mode == 1:
            best_route, best_cost = local_search_mode1(initial_route, customers, travel_time, max_iter=10000)
        elif mode == 2:
            best_route, best_cost = local_search_mode2(initial_route, customers, travel_time, max_iter=10000)
        elif mode == 3:
            best_route, best_cost = local_search_mode3(initial_route, customers, travel_time, max_iter=10000)
        elif mode == 4:
            best_route, best_cost = local_search_mode4(initial_route, customers, travel_time, max_iter=10000)
        elif mode == 5:
            # GRASP doesn't need initial solution
            best_route, best_cost = local_search_mode5(initial_route, customers, travel_time, max_iter=10000)
            break  # GRASP generates its own solutions
        elif mode == 6:
            best_route, best_cost = local_search_mode6(initial_route, customers, travel_time, max_iter=10000)
        else:
            best_route = initial_route
            best_cost = calculate_total_time(best_route, customers, travel_time)
        
        if best_cost < best_overall_cost:
            best_overall_route = best_route
            best_overall_cost = best_cost
    
    return n, best_overall_route

if __name__ == "__main__":
    n, route = solve(MODE)
    print(n)
    print(' '.join(map(str, route)))

