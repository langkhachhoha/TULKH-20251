"""
Ant Colony Optimization (ACO) Algorithm for VRPTW
Mode 1: Basic ACO with pheromone update
Mode 2: Max-Min Ant System (MMAS)
Mode 3: Ant Colony System (ACS) with local pheromone update
Mode 4: Elitist Ant System with best-so-far update
Mode 5: Rank-based Ant System
Mode 6: ACO with local search hybrid
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
        
        if arrival_time > customer['latest']:
            return float('inf')
        
        start_service = max(arrival_time, customer['earliest'])
        current_time = start_service + customer['duration']
        total_travel += travel
        prev = idx
    
    return total_travel

def initialize_pheromone(n, initial_value=1.0):
    """Initialize pheromone matrix"""
    pheromone = [[initial_value for _ in range(n + 1)] for _ in range(n + 1)]
    return pheromone

def calculate_heuristic(customers, travel_time, n):
    """Calculate heuristic information (visibility)"""
    heuristic = [[0.0 for _ in range(n + 1)] for _ in range(n + 1)]
    
    for i in range(n + 1):
        for j in range(1, n + 1):
            if i != j:
                # Combine distance and time window urgency
                distance = travel_time[i][j]
                if distance > 0:
                    # Higher heuristic for closer customers
                    heuristic[i][j] = 1.0 / distance
                else:
                    heuristic[i][j] = 1.0
    
    return heuristic

def select_next_customer(current_pos, current_time, unvisited, pheromone, heuristic, 
                        customers, travel_time, alpha=1.0, beta=2.0, q0=0.0):
    """Select next customer using ACO probability rule"""
    if not unvisited:
        return None
    
    # Calculate probabilities for each unvisited customer
    probabilities = []
    total = 0.0
    
    for customer_id in unvisited:
        travel = travel_time[current_pos][customer_id]
        arrival_time = current_time + travel
        customer = customers[customer_id - 1]
        
        # Check time window feasibility
        if arrival_time > customer['latest']:
            continue
        
        # Calculate attractiveness
        tau = pheromone[current_pos][customer_id] ** alpha
        eta = heuristic[current_pos][customer_id] ** beta
        
        # Add time window urgency factor
        urgency = 1.0
        if customer['latest'] - arrival_time < 50:  # Close to deadline
            urgency = 2.0
        
        attractiveness = tau * eta * urgency
        probabilities.append((customer_id, attractiveness))
        total += attractiveness
    
    if not probabilities:
        # No feasible customer, return random from unvisited
        return random.choice(list(unvisited)) if unvisited else None
    
    # Exploitation vs exploration (for ACS)
    if random.random() < q0:
        # Exploitation: choose best
        return max(probabilities, key=lambda x: x[1])[0]
    
    # Exploration: probabilistic selection
    if total == 0:
        return probabilities[0][0]
    
    # Normalize probabilities
    probabilities = [(cid, attr / total) for cid, attr in probabilities]
    
    # Roulette wheel selection
    r = random.random()
    cumulative = 0.0
    for customer_id, prob in probabilities:
        cumulative += prob
        if r <= cumulative:
            return customer_id
    
    return probabilities[-1][0]

def construct_solution(n, customers, travel_time, pheromone, heuristic, 
                       alpha=1.0, beta=2.0, q0=0.0):
    """Construct a solution using ACO"""
    route = []
    unvisited = set(range(1, n + 1))
    current_pos = 0
    current_time = 0
    
    while unvisited:
        next_customer = select_next_customer(
            current_pos, current_time, unvisited, pheromone, heuristic,
            customers, travel_time, alpha, beta, q0
        )
        
        if next_customer is None:
            # Add remaining customers in random order
            route.extend(list(unvisited))
            break
        
        route.append(next_customer)
        unvisited.remove(next_customer)
        
        # Update position and time
        travel = travel_time[current_pos][next_customer]
        arrival_time = current_time + travel
        customer = customers[next_customer - 1]
        start_service = max(arrival_time, customer['earliest'])
        current_time = start_service + customer['duration']
        current_pos = next_customer
    
    return route

def update_pheromone_basic(pheromone, solutions, evaporation_rate=0.1):
    """Basic pheromone update (Mode 1)"""
    n = len(pheromone) - 1
    
    # Evaporation
    for i in range(n + 1):
        for j in range(n + 1):
            pheromone[i][j] *= (1 - evaporation_rate)
    
    # Add pheromone from all ants
    for route, cost in solutions:
        if cost == float('inf'):
            continue
        
        deposit = 1.0 / cost
        prev = 0
        for customer in route:
            pheromone[prev][customer] += deposit
            prev = customer

def update_pheromone_mmas(pheromone, best_route, best_cost, evaporation_rate=0.1,
                          tau_min=0.01, tau_max=10.0):
    """Max-Min Ant System pheromone update (Mode 2)"""
    n = len(pheromone) - 1
    
    # Evaporation
    for i in range(n + 1):
        for j in range(n + 1):
            pheromone[i][j] *= (1 - evaporation_rate)
    
    # Add pheromone only from best ant
    if best_cost != float('inf'):
        deposit = 1.0 / best_cost
        prev = 0
        for customer in best_route:
            pheromone[prev][customer] += deposit
            prev = customer
    
    # Apply min-max limits
    for i in range(n + 1):
        for j in range(n + 1):
            pheromone[i][j] = max(tau_min, min(tau_max, pheromone[i][j]))

def local_pheromone_update(pheromone, prev, current, xi=0.1, tau0=1.0):
    """Local pheromone update for ACS (Mode 3)"""
    pheromone[prev][current] = (1 - xi) * pheromone[prev][current] + xi * tau0

def update_pheromone_acs(pheromone, best_route, best_cost, evaporation_rate=0.1):
    """Ant Colony System global pheromone update (Mode 3)"""
    n = len(pheromone) - 1
    
    # Global evaporation
    for i in range(n + 1):
        for j in range(n + 1):
            pheromone[i][j] *= (1 - evaporation_rate)
    
    # Add pheromone only on best tour
    if best_cost != float('inf'):
        deposit = 1.0 / best_cost
        prev = 0
        for customer in best_route:
            pheromone[prev][customer] += deposit
            prev = customer

def update_pheromone_elitist(pheromone, solutions, best_ever_route, best_ever_cost,
                             evaporation_rate=0.1, elite_weight=2.0):
    """Elitist Ant System pheromone update (Mode 4)"""
    n = len(pheromone) - 1
    
    # Evaporation
    for i in range(n + 1):
        for j in range(n + 1):
            pheromone[i][j] *= (1 - evaporation_rate)
    
    # Add pheromone from all ants
    for route, cost in solutions:
        if cost == float('inf'):
            continue
        deposit = 1.0 / cost
        prev = 0
        for customer in route:
            pheromone[prev][customer] += deposit
            prev = customer
    
    # Add extra pheromone from best-ever solution
    if best_ever_cost != float('inf'):
        deposit = elite_weight / best_ever_cost
        prev = 0
        for customer in best_ever_route:
            pheromone[prev][customer] += deposit
            prev = customer

def update_pheromone_rank(pheromone, solutions, evaporation_rate=0.1, num_elites=5):
    """Rank-based Ant System pheromone update (Mode 5)"""
    n = len(pheromone) - 1
    
    # Evaporation
    for i in range(n + 1):
        for j in range(n + 1):
            pheromone[i][j] *= (1 - evaporation_rate)
    
    # Sort solutions by cost
    valid_solutions = [(route, cost) for route, cost in solutions if cost != float('inf')]
    valid_solutions.sort(key=lambda x: x[1])
    
    # Add pheromone from top-ranked ants
    for rank, (route, cost) in enumerate(valid_solutions[:num_elites]):
        weight = num_elites - rank
        deposit = weight / cost
        prev = 0
        for customer in route:
            pheromone[prev][customer] += deposit
            prev = customer

def two_opt_improvement(route, customers, travel_time):
    """Apply 2-opt local search improvement"""
    best_route = route[:]
    best_cost = calculate_total_time(best_route, customers, travel_time)
    improved = True
    
    n = len(route)
    max_iterations = min(100, n * 5)
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(indices)
        
        for i, j in indices[:min(len(indices), n * 3)]:
            new_route = best_route[:]
            new_route[i], new_route[j] = new_route[j], new_route[i]
            new_cost = calculate_total_time(new_route, customers, travel_time)
            
            if new_cost < best_cost:
                best_route = new_route
                best_cost = new_cost
                improved = True
                break
    
    return best_route, best_cost

def aco_mode1(n, customers, travel_time, num_ants=20, max_iterations=100):
    """Mode 1: Basic ACO with pheromone update"""
    pheromone = initialize_pheromone(n, initial_value=1.0)
    heuristic = calculate_heuristic(customers, travel_time, n)
    
    best_route = None
    best_cost = float('inf')
    
    alpha = 1.0
    beta = 2.0
    evaporation_rate = 0.1
    
    for iteration in range(max_iterations):
        solutions = []
        
        # Each ant constructs a solution
        for ant in range(num_ants):
            route = construct_solution(n, customers, travel_time, pheromone, heuristic,
                                      alpha, beta, q0=0.0)
            cost = calculate_total_time(route, customers, travel_time)
            solutions.append((route, cost))
            
            if cost < best_cost:
                best_route = route[:]
                best_cost = cost
        
        # Update pheromone
        update_pheromone_basic(pheromone, solutions, evaporation_rate)
    
    return best_route, best_cost

def aco_mode2(n, customers, travel_time, num_ants=20, max_iterations=100):
    """Mode 2: Max-Min Ant System (MMAS)"""
    pheromone = initialize_pheromone(n, initial_value=5.0)
    heuristic = calculate_heuristic(customers, travel_time, n)
    
    best_route = None
    best_cost = float('inf')
    iteration_best_route = None
    iteration_best_cost = float('inf')
    
    alpha = 1.0
    beta = 3.0
    evaporation_rate = 0.05
    tau_min = 0.01
    tau_max = 10.0
    
    for iteration in range(max_iterations):
        iteration_best_route = None
        iteration_best_cost = float('inf')
        
        # Each ant constructs a solution
        for ant in range(num_ants):
            route = construct_solution(n, customers, travel_time, pheromone, heuristic,
                                      alpha, beta, q0=0.0)
            cost = calculate_total_time(route, customers, travel_time)
            
            if cost < iteration_best_cost:
                iteration_best_route = route[:]
                iteration_best_cost = cost
            
            if cost < best_cost:
                best_route = route[:]
                best_cost = cost
        
        # Update pheromone using best-so-far or iteration-best
        if iteration % 5 == 0:
            update_route = best_route
            update_cost = best_cost
        else:
            update_route = iteration_best_route
            update_cost = iteration_best_cost
        
        update_pheromone_mmas(pheromone, update_route, update_cost, 
                             evaporation_rate, tau_min, tau_max)
    
    return best_route, best_cost

def aco_mode3(n, customers, travel_time, num_ants=20, max_iterations=100):
    """Mode 3: Ant Colony System (ACS) with local pheromone update"""
    pheromone = initialize_pheromone(n, initial_value=1.0)
    heuristic = calculate_heuristic(customers, travel_time, n)
    
    best_route = None
    best_cost = float('inf')
    
    alpha = 1.0
    beta = 2.0
    evaporation_rate = 0.1
    q0 = 0.9  # Exploitation probability
    xi = 0.1  # Local evaporation
    
    for iteration in range(max_iterations):
        # Each ant constructs a solution with local pheromone update
        for ant in range(num_ants):
            route = []
            unvisited = set(range(1, n + 1))
            current_pos = 0
            current_time = 0
            
            while unvisited:
                next_customer = select_next_customer(
                    current_pos, current_time, unvisited, pheromone, heuristic,
                    customers, travel_time, alpha, beta, q0
                )
                
                if next_customer is None:
                    route.extend(list(unvisited))
                    break
                
                route.append(next_customer)
                unvisited.remove(next_customer)
                
                # Local pheromone update
                local_pheromone_update(pheromone, current_pos, next_customer, xi)
                
                # Update position and time
                travel = travel_time[current_pos][next_customer]
                arrival_time = current_time + travel
                customer = customers[next_customer - 1]
                start_service = max(arrival_time, customer['earliest'])
                current_time = start_service + customer['duration']
                current_pos = next_customer
            
            cost = calculate_total_time(route, customers, travel_time)
            
            if cost < best_cost:
                best_route = route[:]
                best_cost = cost
        
        # Global pheromone update
        update_pheromone_acs(pheromone, best_route, best_cost, evaporation_rate)
    
    return best_route, best_cost

def aco_mode4(n, customers, travel_time, num_ants=20, max_iterations=100):
    """Mode 4: Elitist Ant System with best-so-far update"""
    pheromone = initialize_pheromone(n, initial_value=1.0)
    heuristic = calculate_heuristic(customers, travel_time, n)
    
    best_route = None
    best_cost = float('inf')
    
    alpha = 1.0
    beta = 2.5
    evaporation_rate = 0.15
    elite_weight = 3.0
    
    for iteration in range(max_iterations):
        solutions = []
        
        # Each ant constructs a solution
        for ant in range(num_ants):
            route = construct_solution(n, customers, travel_time, pheromone, heuristic,
                                      alpha, beta, q0=0.0)
            cost = calculate_total_time(route, customers, travel_time)
            solutions.append((route, cost))
            
            if cost < best_cost:
                best_route = route[:]
                best_cost = cost
        
        # Update pheromone with elitist strategy
        update_pheromone_elitist(pheromone, solutions, best_route, best_cost,
                                evaporation_rate, elite_weight)
    
    return best_route, best_cost

def aco_mode5(n, customers, travel_time, num_ants=20, max_iterations=100):
    """Mode 5: Rank-based Ant System"""
    pheromone = initialize_pheromone(n, initial_value=1.0)
    heuristic = calculate_heuristic(customers, travel_time, n)
    
    best_route = None
    best_cost = float('inf')
    
    alpha = 1.0
    beta = 2.0
    evaporation_rate = 0.1
    num_elites = min(5, num_ants // 2)
    
    for iteration in range(max_iterations):
        solutions = []
        
        # Each ant constructs a solution
        for ant in range(num_ants):
            route = construct_solution(n, customers, travel_time, pheromone, heuristic,
                                      alpha, beta, q0=0.0)
            cost = calculate_total_time(route, customers, travel_time)
            solutions.append((route, cost))
            
            if cost < best_cost:
                best_route = route[:]
                best_cost = cost
        
        # Update pheromone using rank-based strategy
        update_pheromone_rank(pheromone, solutions, evaporation_rate, num_elites)
    
    return best_route, best_cost

def aco_mode6(n, customers, travel_time, num_ants=20, max_iterations=100):
    """Mode 6: ACO with local search hybrid"""
    pheromone = initialize_pheromone(n, initial_value=1.0)
    heuristic = calculate_heuristic(customers, travel_time, n)
    
    best_route = None
    best_cost = float('inf')
    
    alpha = 1.0
    beta = 2.0
    evaporation_rate = 0.1
    
    for iteration in range(max_iterations):
        solutions = []
        
        # Each ant constructs a solution
        for ant in range(num_ants):
            route = construct_solution(n, customers, travel_time, pheromone, heuristic,
                                      alpha, beta, q0=0.0)
            
            # Apply local search to some ants
            if random.random() < 0.5:  # 50% of ants use local search
                route, cost = two_opt_improvement(route, customers, travel_time)
            else:
                cost = calculate_total_time(route, customers, travel_time)
            
            solutions.append((route, cost))
            
            if cost < best_cost:
                best_route = route[:]
                best_cost = cost
        
        # Update pheromone
        update_pheromone_basic(pheromone, solutions, evaporation_rate)
        
        # Apply local search to best solution periodically
        if iteration % 10 == 0 and best_route is not None:
            improved_route, improved_cost = two_opt_improvement(best_route, customers, travel_time)
            if improved_cost < best_cost:
                best_route = improved_route
                best_cost = improved_cost
    
    return best_route, best_cost

def solve(mode):
    n, customers, travel_time = read_input()
    
    # ACO parameters
    num_ants = min(20, n)
    max_iterations = 150
    
    # Run ACO based on mode
    if mode == 1:
        best_route, best_cost = aco_mode1(n, customers, travel_time, num_ants, max_iterations)
    elif mode == 2:
        best_route, best_cost = aco_mode2(n, customers, travel_time, num_ants, max_iterations)
    elif mode == 3:
        best_route, best_cost = aco_mode3(n, customers, travel_time, num_ants, max_iterations)
    elif mode == 4:
        best_route, best_cost = aco_mode4(n, customers, travel_time, num_ants, max_iterations)
    elif mode == 5:
        best_route, best_cost = aco_mode5(n, customers, travel_time, num_ants, max_iterations)
    elif mode == 6:
        best_route, best_cost = aco_mode6(n, customers, travel_time, num_ants, max_iterations)
    else:
        # Default: use mode 1
        best_route, best_cost = aco_mode1(n, customers, travel_time, num_ants, max_iterations)
    
    return n, best_route

if __name__ == "__main__":
    n, route = solve(MODE)
    print(n)
    print(' '.join(map(str, route)))

