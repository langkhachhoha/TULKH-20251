"""
Improved Tabu Search Algorithm for VRPTW
Mode 1: Reactive tabu search with intensification/diversification
Mode 2: Robust tabu search with strategic oscillation
Mode 3: Adaptive tabu search with aspiration plus
Mode 4: Tabu search with path relinking
Mode 5: Granular tabu search with candidate list
Mode 6: Probabilistic tabu search with threshold accepting
"""

import random
import sys
import copy

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
        
        if arrival_time > customer['latest']:
            return float('inf')
        
        start_service = max(arrival_time, customer['earliest'])
        current_time = start_service + customer['duration']
        total_travel += travel
        prev = idx
    
    return total_travel

def generate_initial_solution(n, customers, travel_time, t0=0, strategy='nearest'):
    """Generate initial feasible solution with different strategies"""
    if strategy == 'nearest':
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
        route = sorted(range(1, n + 1), key=lambda x: customers[x-1]['earliest'])
    else:
        route = list(range(1, n + 1))
        random.shuffle(route)
    
    return route

def two_opt_swap(route, i, j):
    """Swap positions i and j"""
    new_route = route[:]
    new_route[i], new_route[j] = new_route[j], new_route[i]
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

def get_neighborhood_candidates(route, n, max_candidates=None):
    """Generate neighborhood with optional candidate list restriction"""
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            candidates.append(('swap', i, j))
    if max_candidates and len(candidates) > max_candidates:
        random.shuffle(candidates)
        candidates = candidates[:max_candidates]
    return candidates

def tabu_search_mode1(route, customers, travel_time, max_iter=2000):
    """Mode 1: Reactive tabu search with intensification/diversification"""
    best_route = route[:]
    best_cost = calculate_total_time(best_route, customers, travel_time)
    current_route = route[:]
    current_cost = best_cost
    
    tabu_list = []
    tabu_tenure = 7
    frequency = {}
    
    no_improve = 0
    phase = 'intensification'  # or 'diversification'
    
    n = len(current_route)
    
    for iteration in range(max_iter):
        # Adjust tabu tenure reactively
        if no_improve > 50:
            tabu_tenure = min(15, tabu_tenure + 1)
        elif no_improve > 100:
            phase = 'diversification'
            tabu_tenure = 3
        
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None
        
        # Generate candidate moves
        candidates = get_neighborhood_candidates(current_route, n, max_candidates=n*3)
        
        for move_type, i, j in candidates:
            move = (i, j)
            
            # Skip tabu moves unless aspiration
            if move in tabu_list:
                continue
            
            neighbor = two_opt_swap(current_route, i, j)
            neighbor_cost = calculate_total_time(neighbor, customers, travel_time)
            
            # In diversification phase, penalize frequent moves
            if phase == 'diversification':
                penalty = frequency.get(move, 0) * 0.5
                neighbor_cost += penalty
            
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost
                best_move = move
        
        if best_neighbor is None:
            break
        
        # Calculate actual cost without penalty
        actual_cost = calculate_total_time(best_neighbor, customers, travel_time)
        current_route = best_neighbor
        current_cost = actual_cost
        
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        frequency[best_move] = frequency.get(best_move, 0) + 1
        
        if actual_cost < best_cost:
            best_route = current_route[:]
            best_cost = actual_cost
            no_improve = 0
            phase = 'intensification'
            tabu_tenure = 7
        else:
            no_improve += 1
    
    return best_route, best_cost

def tabu_search_mode2(route, customers, travel_time, max_iter=2000):
    """Mode 2: Robust tabu search with strategic oscillation"""
    best_route = route[:]
    best_cost = calculate_total_time(best_route, customers, travel_time)
    current_route = route[:]
    
    tabu_list = []
    tabu_tenure = 10
    
    n = len(current_route)
    oscillation_strength = 0
    
    for iteration in range(max_iter):
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None
        best_move_type = None
        
        # Alternate between different move types (strategic oscillation)
        if iteration % 3 == 0:
            move_types = [('swap', two_opt_swap)]
        elif iteration % 3 == 1:
            move_types = [('insert', insert_move)]
        else:
            move_types = [('reverse', reverse_segment)]
        
        for move_name, move_func in move_types:
            if move_name == 'swap':
                candidates = [(i, j) for i in range(n) for j in range(i + 1, n)]
            elif move_name == 'insert':
                candidates = [(i, j) for i in range(n) for j in range(n) if i != j]
            else:
                candidates = [(i, j) for i in range(n) for j in range(i + 2, n)]
            
            random.shuffle(candidates)
            candidates = candidates[:min(len(candidates), n * 4)]
            
            for i, j in candidates:
                move = (move_name, i, j)
                
                if move in tabu_list:
                    continue
                
                neighbor = move_func(current_route, i, j)
                neighbor_cost = calculate_total_time(neighbor, customers, travel_time)
                
                # Add oscillation: sometimes accept slightly worse
                adjusted_cost = neighbor_cost - oscillation_strength
                
                if adjusted_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = adjusted_cost
                    best_move = move
                    best_move_type = move_name
        
        if best_neighbor is None:
            oscillation_strength += 5  # Increase oscillation
            if oscillation_strength > 20:
                oscillation_strength = 0
            continue
        
        actual_cost = calculate_total_time(best_neighbor, customers, travel_time)
        current_route = best_neighbor
        
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        if actual_cost < best_cost:
            best_route = current_route[:]
            best_cost = actual_cost
            oscillation_strength = 0
        else:
            oscillation_strength = min(oscillation_strength + 1, 10)
    
    return best_route, best_cost

def tabu_search_mode3(route, customers, travel_time, max_iter=2000):
    """Mode 3: Adaptive tabu search with aspiration plus"""
    best_route = route[:]
    best_cost = calculate_total_time(best_route, customers, travel_time)
    current_route = route[:]
    
    tabu_list = {}  # Move -> iteration when it becomes non-tabu
    base_tenure = 8
    
    n = len(current_route)
    best_iteration = 0
    
    for iteration in range(max_iter):
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None
        
        # Adaptive tabu tenure based on progress
        progress_ratio = (iteration - best_iteration) / max(1, iteration)
        tabu_tenure = int(base_tenure * (1 + progress_ratio * 2))
        
        # Try multiple move types
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                candidates.append(('swap', i, j))
        for i in range(n):
            for j in range(n):
                if i != j and abs(i - j) > 1:
                    candidates.append(('insert', i, j))
        
        random.shuffle(candidates)
        candidates = candidates[:min(len(candidates), n * 5)]
        
        for move_type, i, j in candidates:
            move = (move_type, i, j)
            
            # Check if move is tabu
            is_tabu = move in tabu_list and tabu_list[move] > iteration
            
            if move_type == 'swap':
                neighbor = two_opt_swap(current_route, i, j)
            else:
                neighbor = insert_move(current_route, i, j)
            
            neighbor_cost = calculate_total_time(neighbor, customers, travel_time)
            
            # Aspiration plus: accept tabu if better than best or significantly better
            if is_tabu:
                current_cost = calculate_total_time(current_route, customers, travel_time)
                if neighbor_cost >= best_cost and neighbor_cost >= current_cost - 5:
                    continue
            
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost
                best_move = move
        
        if best_neighbor is None:
            # Restart from best with perturbation
            current_route = best_route[:]
            for _ in range(3):
                i, j = random.sample(range(n), 2)
                current_route[i], current_route[j] = current_route[j], current_route[i]
            continue
        
        current_route = best_neighbor
        
        # Update tabu list
        tabu_list[best_move] = iteration + tabu_tenure
        
        if best_neighbor_cost < best_cost:
            best_route = current_route[:]
            best_cost = best_neighbor_cost
            best_iteration = iteration
    
    return best_route, best_cost

def path_relink(route1, route2, customers, travel_time):
    """Path relinking between two solutions"""
    current = route1[:]
    best = route1[:]
    best_cost = calculate_total_time(best, customers, travel_time)
    
    # Find differences
    n = len(route1)
    max_moves = min(n // 2, 10)
    
    for _ in range(max_moves):
        best_move_cost = float('inf')
        best_move = None
        
        # Try to make current more similar to route2
        for i in range(n):
            if current[i] != route2[i]:
                # Find where route2[i] is in current
                j = current.index(route2[i])
                if i != j:
                    # Try swap
                    neighbor = current[:]
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    cost = calculate_total_time(neighbor, customers, travel_time)
                    
                    if cost < best_move_cost:
                        best_move_cost = cost
                        best_move = neighbor
        
        if best_move is None:
            break
        
        current = best_move
        if best_move_cost < best_cost:
            best = current[:]
            best_cost = best_move_cost
    
    return best, best_cost

def tabu_search_mode4(route, customers, travel_time, max_iter=2000):
    """Mode 4: Tabu search with path relinking"""
    best_route = route[:]
    best_cost = calculate_total_time(best_route, customers, travel_time)
    current_route = route[:]
    
    tabu_list = []
    tabu_tenure = 10
    elite_solutions = [best_route[:]]
    max_elite = 5
    
    n = len(current_route)
    
    for iteration in range(max_iter):
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None
        
        # Regular tabu search
        candidates = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(candidates)
        candidates = candidates[:min(len(candidates), n * 4)]
        
        for i, j in candidates:
            move = (i, j)
            
            if move in tabu_list:
                continue
            
            neighbor = two_opt_swap(current_route, i, j)
            neighbor_cost = calculate_total_time(neighbor, customers, travel_time)
            
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost
                best_move = move
        
        if best_neighbor is None:
            # Path relink with elite solution
            if len(elite_solutions) > 1:
                target = random.choice(elite_solutions)
                relinked, relinked_cost = path_relink(current_route, target, customers, travel_time)
                if relinked_cost < best_cost:
                    best_route = relinked[:]
                    best_cost = relinked_cost
            current_route = best_route[:]
            continue
        
        current_route = best_neighbor
        
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        if best_neighbor_cost < best_cost:
            best_route = current_route[:]
            best_cost = best_neighbor_cost
            
            # Update elite solutions
            elite_solutions.append(best_route[:])
            if len(elite_solutions) > max_elite:
                # Remove worst
                elite_solutions.sort(key=lambda x: calculate_total_time(x, customers, travel_time))
                elite_solutions = elite_solutions[:max_elite]
        
        # Periodic path relinking
        if iteration % 200 == 0 and len(elite_solutions) > 1:
            for elite in elite_solutions[:3]:
                relinked, relinked_cost = path_relink(best_route, elite, customers, travel_time)
                if relinked_cost < best_cost:
                    best_route = relinked[:]
                    best_cost = relinked_cost
    
    return best_route, best_cost

def tabu_search_mode5(route, customers, travel_time, max_iter=2000):
    """Mode 5: Granular tabu search with candidate list"""
    best_route = route[:]
    best_cost = calculate_total_time(best_route, customers, travel_time)
    current_route = route[:]
    
    tabu_list = []
    tabu_tenure = 12
    frequency = {}
    
    n = len(current_route)
    
    # Build granular candidate list (restrict to nearby customers)
    # Customers close in time window or distance
    def build_granular_list(route, customers, travel_time):
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = route[i], route[j]
                # Add if close in time or distance
                time_diff = abs(customers[c1-1]['earliest'] - customers[c2-1]['earliest'])
                distance = travel_time[c1][c2]
                
                if time_diff < 100 or distance < 50 or random.random() < 0.3:
                    candidates.append((i, j))
        return candidates
    
    for iteration in range(max_iter):
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None
        
        # Rebuild granular list periodically
        if iteration % 100 == 0:
            granular_candidates = build_granular_list(current_route, customers, travel_time)
        
        # Combine swap, insert, and reverse moves
        move_candidates = []
        
        # Use granular list for swaps
        for i, j in granular_candidates[:n * 4]:
            move_candidates.append(('swap', i, j))
        
        # Add some insert moves
        for _ in range(n * 2):
            i, j = random.sample(range(n), 2)
            move_candidates.append(('insert', i, j))
        
        # Add some reverse moves
        for _ in range(n):
            i = random.randint(0, n - 3)
            j = random.randint(i + 2, n - 1)
            move_candidates.append(('reverse', i, j))
        
        random.shuffle(move_candidates)
        
        for move_type, i, j in move_candidates:
            move = (move_type, i, j)
            
            # Check tabu with aspiration
            is_tabu = move in tabu_list
            
            if move_type == 'swap':
                neighbor = two_opt_swap(current_route, i, j)
            elif move_type == 'insert':
                neighbor = insert_move(current_route, i, j)
            else:
                neighbor = reverse_segment(current_route, i, j)
            
            neighbor_cost = calculate_total_time(neighbor, customers, travel_time)
            
            # Aspiration criterion
            if is_tabu and neighbor_cost >= best_cost:
                continue
            
            # Penalize frequently used moves slightly
            penalty = frequency.get(move, 0) * 0.05
            adjusted_cost = neighbor_cost + penalty
            
            if adjusted_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = adjusted_cost
                best_move = move
        
        if best_neighbor is None:
            # Perturbation
            current_route = best_route[:]
            for _ in range(5):
                i, j = random.sample(range(n), 2)
                current_route[i], current_route[j] = current_route[j], current_route[i]
            continue
        
        actual_cost = calculate_total_time(best_neighbor, customers, travel_time)
        current_route = best_neighbor
        
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        frequency[best_move] = frequency.get(best_move, 0) + 1
        
        if actual_cost < best_cost:
            best_route = current_route[:]
            best_cost = actual_cost
            frequency = {}  # Reset frequency on improvement
    
    return best_route, best_cost

def tabu_search_mode6(route, customers, travel_time, max_iter=2000):
    """Mode 6: Probabilistic tabu search with threshold accepting"""
    best_route = route[:]
    best_cost = calculate_total_time(best_route, customers, travel_time)
    current_route = route[:]
    
    tabu_list = []
    tabu_tenure = 8
    
    n = len(current_route)
    
    # Threshold accepting parameters
    threshold = best_cost * 0.05  # Allow 5% worse solutions initially
    threshold_decay = 0.995
    
    for iteration in range(max_iter):
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None
        
        # Generate multiple types of moves
        candidates = []
        
        # Swap moves
        for i in range(n):
            for j in range(i + 1, n):
                candidates.append(('swap', i, j))
        
        # Insert moves (subset)
        for _ in range(n * 2):
            i, j = random.sample(range(n), 2)
            candidates.append(('insert', i, j))
        
        random.shuffle(candidates)
        candidates = candidates[:min(len(candidates), n * 5)]
        
        for move_type, i, j in candidates:
            move = (move_type, i, j)
            
            # Probabilistic tabu: sometimes ignore tabu
            if move in tabu_list and random.random() > 0.15:
                continue
            
            if move_type == 'swap':
                neighbor = two_opt_swap(current_route, i, j)
            else:
                neighbor = insert_move(current_route, i, j)
            
            neighbor_cost = calculate_total_time(neighbor, customers, travel_time)
            
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost
                best_move = move
        
        if best_neighbor is None:
            # Perturbation
            current_route = best_route[:]
            for _ in range(3):
                i, j = random.sample(range(n), 2)
                current_route[i], current_route[j] = current_route[j], current_route[i]
            threshold = best_cost * 0.05
            continue
        
        current_cost = calculate_total_time(current_route, customers, travel_time)
        
        # Threshold accepting: accept if within threshold
        if best_neighbor_cost <= current_cost + threshold:
            current_route = best_neighbor
            
            tabu_list.append(best_move)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
            
            if best_neighbor_cost < best_cost:
                best_route = current_route[:]
                best_cost = best_neighbor_cost
                threshold = best_cost * 0.05  # Reset threshold
        
        # Decay threshold
        threshold *= threshold_decay
        threshold = max(threshold, 1.0)  # Minimum threshold
    
    return best_route, best_cost

def solve(mode):
    n, customers, travel_time = read_input()
    
    # Try multiple initial solutions
    best_overall_route = None
    best_overall_cost = float('inf')
    
    strategies = ['nearest', 'earliest', 'random']
    
    for strategy in strategies[:2]:  # Use first 2 to save time
        initial_route = generate_initial_solution(n, customers, travel_time, strategy=strategy)
        
        if mode == 1:
            best_route, best_cost = tabu_search_mode1(initial_route, customers, travel_time)
        elif mode == 2:
            best_route, best_cost = tabu_search_mode2(initial_route, customers, travel_time)
        elif mode == 3:
            best_route, best_cost = tabu_search_mode3(initial_route, customers, travel_time)
        elif mode == 4:
            best_route, best_cost = tabu_search_mode4(initial_route, customers, travel_time)
        elif mode == 5:
            best_route, best_cost = tabu_search_mode5(initial_route, customers, travel_time)
        elif mode == 6:
            best_route, best_cost = tabu_search_mode6(initial_route, customers, travel_time)
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

