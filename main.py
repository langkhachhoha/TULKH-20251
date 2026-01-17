"""
Main script to run all optimization methods with time constraints
Tests: N5, N10, N100, N200, N300, N500, N600, N700, N900, N1000
Time limits:
- N5, N10, N100: 60s
- N200, N300, N500: 120s
- N600, N700, N900, N1000: 180s
"""

import os
import sys
import time
import signal
import random
import re
from typing import List, Tuple, Optional
import multiprocessing
from multiprocessing import Process, Queue

# Import from local_search
from local_search import (
    read_input,
    calculate_total_time,
    generate_initial_solution,
    is_feasible,
    local_search_mode1,
    local_search_mode2,
    local_search_mode3,
    local_search_mode4,
    local_search_mode5,
    local_search_mode6
)

# Import from tabu_search
from tabu_search import (
    tabu_search_mode1,
    tabu_search_mode2,
    tabu_search_mode3
)

# Import from GA
from GA import GA_VRPTW

# Import exact algorithms
import subprocess
import heapq


def read_input_from_file(filepath: str):
    """Read input from file"""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    n = int(lines[0])
    
    customers = []
    for i in range(1, n + 1):
        e, l, d = map(int, lines[i].split())
        customers.append({'id': i, 'earliest': e, 'latest': l, 'duration': d})
    
    travel_time = []
    for i in range(n + 1, n + 2 + n):
        row = list(map(int, lines[i].split()))
        travel_time.append(row)
    
    return n, customers, travel_time


def run_with_timeout(func, args, timeout, result_queue):
    """Run a function with timeout using multiprocessing"""
    def wrapper():
        try:
            result = func(*args)
            result_queue.put(('success', result))
        except Exception as e:
            result_queue.put(('error', str(e)))
    
    process = Process(target=wrapper)
    process.start()
    process.join(timeout)
    
    if process.is_alive():
        # Timeout occurred
        process.terminate()
        process.join()
        return None
    
    # Get result
    if not result_queue.empty():
        status, result = result_queue.get()
        if status == 'success':
            return result
    
    return None


def run_local_search_mode(mode_num, n, customers, travel_time, time_limit):
    """Run a local search mode with time limit"""
    start_time = time.time()
    
    # Generate initial solution
    best_overall_route = None
    best_overall_cost = float('inf')
    
    strategies = ['nearest', 'earliest', 'latest']
    
    for strategy in strategies:
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break
        
        initial_route = generate_initial_solution(n, customers, travel_time, strategy=strategy)
        
        # Skip if initial solution is not feasible
        if not is_feasible(initial_route, customers, travel_time):
            continue
        
        # Calculate remaining time
        remaining_time = time_limit - elapsed
        if remaining_time <= 0:
            break
        
        # Adjust max_iter based on remaining time
        max_iter = int(10000 * (remaining_time / time_limit))
        
        try:
            if mode_num == 1:
                best_route, best_cost = local_search_mode1(initial_route, customers, travel_time, max_iter=max_iter)
            elif mode_num == 2:
                best_route, best_cost = local_search_mode2(initial_route, customers, travel_time, max_iter=max_iter)
            elif mode_num == 3:
                best_route, best_cost = local_search_mode3(initial_route, customers, travel_time, max_iter=max_iter)
            elif mode_num == 4:
                best_route, best_cost = local_search_mode4(initial_route, customers, travel_time, max_iter=max_iter)
            elif mode_num == 5:
                best_route, best_cost = local_search_mode5(initial_route, customers, travel_time, max_iter=max_iter)
                break  # GRASP generates its own solutions
            elif mode_num == 6:
                best_route, best_cost = local_search_mode6(initial_route, customers, travel_time, max_iter=max_iter)
            else:
                best_route = initial_route
                best_cost = calculate_total_time(best_route, customers, travel_time)
            
            # Check if solution is feasible
            if is_feasible(best_route, customers, travel_time) and best_cost != float('inf'):
                if best_cost < best_overall_cost:
                    best_overall_route = best_route
                    best_overall_cost = best_cost
        except Exception as e:
            print(f"    Error in strategy {strategy}: {e}")
            continue
        
        # Check time limit
        if time.time() - start_time >= time_limit:
            break
    
    # If no solution found, try to generate a feasible one
    if best_overall_route is None or best_overall_cost == float('inf'):
        # Try all strategies to find a feasible solution
        for strategy in ['earliest', 'latest', 'nearest']:
            route = generate_initial_solution(n, customers, travel_time, strategy=strategy)
            if is_feasible(route, customers, travel_time):
                cost = calculate_total_time(route, customers, travel_time)
                if cost < best_overall_cost:
                    best_overall_route = route
                    best_overall_cost = cost
                    break
    
    return best_overall_route, best_overall_cost


def run_tabu_search_mode(mode_num, n, customers, travel_time, time_limit):
    """Run a tabu search mode with time limit"""
    start_time = time.time()
    
    best_overall_route = None
    best_overall_cost = float('inf')
    
    strategies = ['nearest', 'earliest']
    
    for strategy in strategies:
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break
        
        initial_route = generate_initial_solution(n, customers, travel_time, strategy=strategy)
        
        # Skip if initial solution is not feasible
        if not is_feasible(initial_route, customers, travel_time):
            continue
        
        # Calculate remaining time
        remaining_time = time_limit - elapsed
        if remaining_time <= 0:
            break
        
        # Adjust max_iter based on remaining time
        max_iter = int(2000 * (remaining_time / time_limit))
        
        try:
            if mode_num == 1:
                best_route, best_cost = tabu_search_mode1(initial_route, customers, travel_time, max_iter=max_iter)
            elif mode_num == 2:
                best_route, best_cost = tabu_search_mode2(initial_route, customers, travel_time, max_iter=max_iter)
            elif mode_num == 3:
                best_route, best_cost = tabu_search_mode3(initial_route, customers, travel_time, max_iter=max_iter)
            else:
                best_route = initial_route
                best_cost = calculate_total_time(best_route, customers, travel_time)
            
            # Check if solution is feasible
            if is_feasible(best_route, customers, travel_time) and best_cost != float('inf'):
                if best_cost < best_overall_cost:
                    best_overall_route = best_route
                    best_overall_cost = best_cost
        except Exception as e:
            print(f"    Error in strategy {strategy}: {e}")
            continue
        
        # Check time limit
        if time.time() - start_time >= time_limit:
            break
    
    # If no solution found, try to generate a feasible one
    if best_overall_route is None or best_overall_cost == float('inf'):
        # Try all strategies to find a feasible solution
        for strategy in ['earliest', 'latest', 'nearest']:
            route = generate_initial_solution(n, customers, travel_time, strategy=strategy)
            if is_feasible(route, customers, travel_time):
                cost = calculate_total_time(route, customers, travel_time)
                if cost < best_overall_cost:
                    best_overall_route = route
                    best_overall_cost = cost
                    break
    
    return best_overall_route, best_overall_cost


# ============= EXACT ALGORITHMS =============

def backtracking_solve(n, customers, travel_time, time_limit):
    """Backtracking algorithm with pruning"""
    start_time = time.time()
    best_cost = float('inf')
    best_route = None
    nodes_explored = [0]
    
    def is_feasible_partial(route, current_time, current_pos):
        """Check if can continue from current state"""
        return current_time >= 0  # Simple check
    
    def lower_bound_remaining(remaining, current_pos):
        """Calculate lower bound for remaining customers"""
        if not remaining:
            return 0
        return min(travel_time[current_pos][c] for c in remaining)
    
    def backtrack(current_route, remaining, current_time, current_pos, current_cost):
        nonlocal best_cost, best_route
        
        # Check time limit
        if time.time() - start_time > time_limit:
            return
        
        nodes_explored[0] += 1
        
        # Base case: all customers visited
        if not remaining:
            if current_cost < best_cost:
                best_cost = current_cost
                best_route = current_route[:]
            return
        
        # Pruning: if current cost + lower bound >= best cost, skip
        lb = lower_bound_remaining(remaining, current_pos)
        if current_cost + lb >= best_cost:
            return
        
        # Try each remaining customer
        for customer_id in sorted(remaining):
            if time.time() - start_time > time_limit:
                return
            
            travel = travel_time[current_pos][customer_id]
            arrival_time = current_time + travel
            customer = customers[customer_id - 1]
            
            # Check feasibility
            if arrival_time > customer['latest']:
                continue
            
            # Calculate new state
            start_service = max(arrival_time, customer['earliest'])
            new_time = start_service + customer['duration']
            new_cost = current_cost + travel
            
            # Pruning: skip if already worse than best
            if new_cost >= best_cost:
                continue
            
            # Recursive call
            new_route = current_route + [customer_id]
            new_remaining = remaining - {customer_id}
            
            backtrack(new_route, new_remaining, new_time, customer_id, new_cost)
    
    # Start backtracking
    remaining = set(range(1, n + 1))
    backtrack([], remaining, 0, 0, 0)
    
    elapsed = time.time() - start_time
    
    if best_route is None:
        return None, "TLE", nodes_explored[0], elapsed
    
    return best_route, best_cost, nodes_explored[0], elapsed


def branch_and_bound_solve(n, customers, travel_time, time_limit):
    """Branch and Bound algorithm using priority queue"""
    start_time = time.time()
    best_cost = float('inf')
    best_route = None
    nodes_explored = 0
    nodes_pruned = 0
    
    def calculate_lower_bound(current_cost, remaining, current_pos):
        """Calculate lower bound for the remaining problem"""
        if not remaining:
            return current_cost
        
        # Minimum cost to reach any remaining customer
        min_to_reach = min(travel_time[current_pos][c] for c in remaining)
        
        # MST approximation for remaining customers
        remaining_list = list(remaining)
        mst_cost = 0
        
        if len(remaining_list) > 1:
            for i in range(len(remaining_list) - 1):
                min_edge = float('inf')
                for j in range(i + 1, len(remaining_list)):
                    edge_cost = travel_time[remaining_list[i]][remaining_list[j]]
                    min_edge = min(min_edge, edge_cost)
                mst_cost += min_edge
        
        return current_cost + min_to_reach + mst_cost
    
    class Node:
        def __init__(self, route, remaining, current_time, current_pos, cost, lower_bound):
            self.route = route
            self.remaining = remaining
            self.current_time = current_time
            self.current_pos = current_pos
            self.cost = cost
            self.lower_bound = lower_bound
        
        def __lt__(self, other):
            if self.lower_bound != other.lower_bound:
                return self.lower_bound < other.lower_bound
            return self.cost < other.cost
    
    # Initialize
    initial_remaining = set(range(1, n + 1))
    initial_lb = calculate_lower_bound(0, initial_remaining, 0)
    initial_node = Node([], initial_remaining, 0, 0, 0, initial_lb)
    
    pq = [initial_node]
    
    while pq and (time.time() - start_time) < time_limit:
        node = heapq.heappop(pq)
        
        nodes_explored += 1
        
        # Pruning
        if node.lower_bound >= best_cost:
            nodes_pruned += 1
            continue
        
        # Complete solution
        if not node.remaining:
            if node.cost < best_cost:
                best_cost = node.cost
                best_route = node.route[:]
            continue
        
        # Branch
        for customer_id in node.remaining:
            if time.time() - start_time >= time_limit:
                break
            
            travel = travel_time[node.current_pos][customer_id]
            arrival_time = node.current_time + travel
            customer = customers[customer_id - 1]
            
            # Check feasibility
            if arrival_time > customer['latest']:
                continue
            
            # Calculate new state
            start_service = max(arrival_time, customer['earliest'])
            new_time = start_service + customer['duration']
            new_cost = node.cost + travel
            
            # Pruning
            if new_cost >= best_cost:
                nodes_pruned += 1
                continue
            
            # Create new node
            new_route = node.route + [customer_id]
            new_remaining = node.remaining - {customer_id}
            new_lb = calculate_lower_bound(new_cost, new_remaining, customer_id)
            
            # Pruning
            if new_lb >= best_cost:
                nodes_pruned += 1
                continue
            
            new_node = Node(new_route, new_remaining, new_time, customer_id, new_cost, new_lb)
            heapq.heappush(pq, new_node)
    
    elapsed = time.time() - start_time
    
    if best_route is None:
        return None, "TLE", nodes_explored, nodes_pruned, elapsed
    
    return best_route, best_cost, nodes_explored, nodes_pruned, elapsed


def ortools_solve(n, customers, travel_time, time_limit):
    """OR-Tools solver"""
    try:
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
    except ImportError:
        return None, "ERROR (OR-Tools not installed)", 0, 0
    
    start_time = time.time()
    
    # Create manager and routing
    manager = pywrapcp.RoutingIndexManager(n + 1, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    # Distance callback (for cost) - don't count return to depot
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if to_node == 0:  # Don't count return to depot
            return 0
        return travel_time[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Time callback (for time windows)
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel = travel_time[from_node][to_node]
        service_time = customers[from_node - 1]['duration'] if from_node > 0 else 0
        return travel + service_time
    
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # Add time dimension
    routing.AddDimension(
        time_callback_index,
        100000,  # allow waiting
        100000,  # max time
        False,
        'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')
    
    # Set time windows
    for customer_idx in range(1, n + 1):
        index = manager.NodeToIndex(customer_idx)
        customer = customers[customer_idx - 1]
        time_dimension.CumulVar(index).SetRange(customer['earliest'], customer['latest'])
    
    depot_idx = manager.NodeToIndex(0)
    time_dimension.CumulVar(depot_idx).SetRange(0, 0)
    
    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = int(time_limit)
    search_parameters.log_search = False
    
    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    
    elapsed = time.time() - start_time
    
    if solution:
        # Extract route
        route = []
        index = routing.Start(0)
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                route.append(node)
            index = solution.Value(routing.NextVar(index))
        
        cost = int(solution.ObjectiveValue())
        return route, cost, 0, elapsed
    else:
        return None, "TLE", 0, elapsed


def run_ga(n, customers, travel_time, time_limit):
    """Run GA with time limit"""
    start_time = time.time()
    
    # Convert format for GA
    time_windows = [(c['earliest'], c['latest']) for c in customers]
    service_times = [c['duration'] for c in customers]
    
    # Adjust population size and generations based on problem size and time limit
    if n <= 10:
        pop_size = 50
        max_gen = 200
    elif n <= 100:
        pop_size = 100
        max_gen = 100
    elif n <= 300:
        pop_size = 80
        max_gen = 50
    else:
        pop_size = 60
        max_gen = 30
    
    ga = GA_VRPTW(
        n=n,
        time_windows=time_windows,
        service_times=service_times,
        travel_matrix=travel_time,
        pop_size=pop_size,
        max_gen=max_gen,
        crossover_rate=0.9,
        mutation_rate=0.2
    )
    
    # Initialize population
    ga.initialize_population()
    
    # Find best initial solution
    for sol in ga.population:
        travel, penalty = ga.calculate_cost(sol)
        cost = travel + penalty
        if cost < ga.best_cost:
            ga.best_cost = cost
            ga.best_solution = sol.copy()
    
    # Evolution with time limit
    gen = 0
    while gen < max_gen and (time.time() - start_time) < time_limit:
        new_population = []
        
        # Elitism
        sorted_pop = sorted(ga.population, key=ga.fitness, reverse=True)
        elite_size = max(2, pop_size // 20)
        new_population.extend([sol.copy() for sol in sorted_pop[:elite_size]])
        
        # Create offspring
        while len(new_population) < pop_size and (time.time() - start_time) < time_limit:
            parent1 = ga.tournament_selection()
            parent2 = ga.tournament_selection()
            
            if random.random() < ga.crossover_rate:
                child = ga.crossover_OX(parent1, parent2)
            else:
                child = parent1.copy() if random.random() < 0.5 else parent2.copy()
            
            if random.random() < ga.mutation_rate:
                if random.random() < 0.5:
                    child = ga.mutation_swap(child)
                else:
                    child = ga.mutation_inversion(child)
            
            new_population.append(child)
        
        ga.population = new_population
        
        # Update best solution
        for sol in ga.population:
            travel, penalty = ga.calculate_cost(sol)
            cost = travel + penalty
            if cost < ga.best_cost:
                ga.best_cost = cost
                ga.best_solution = sol.copy()
        
        # Local search periodically
        if gen % 10 == 0 and ga.best_solution and (time.time() - start_time) < time_limit - 5:
            try:
                improved = ga.local_search_2opt(ga.best_solution)
                travel, penalty = ga.calculate_cost(improved)
                cost = travel + penalty
                if cost < ga.best_cost:
                    ga.best_cost = cost
                    ga.best_solution = improved.copy()
            except:
                pass
        
        gen += 1
    
    # Return best feasible solution
    if ga.best_solution and ga.is_feasible(ga.best_solution):
        travel, penalty = ga.calculate_cost(ga.best_solution)
        return ga.best_solution, travel
    
    # Try to find a feasible solution in population
    for sol in ga.population:
        if ga.is_feasible(sol):
            travel, penalty = ga.calculate_cost(sol)
            return sol, travel
    
    # Fallback: generate a simple feasible solution
    fallback = generate_initial_solution(n, customers, travel_time, strategy='nearest')
    fallback_cost = calculate_total_time(fallback, customers, travel_time)
    return fallback, fallback_cost


def get_time_limit(n):
    """Get time limit based on problem size"""
    if n <= 100:
        return 60
    elif n <= 500:
        return 120
    else:
        return 180


def run_method(method_name, method_func, time_limit):
    """Run a method with time limit and return result"""
    print(f"  Running {method_name}...", end=' ', flush=True)
    start_time = time.time()
    
    try:
        best_route, best_cost = method_func(time_limit)
        elapsed = time.time() - start_time
        print(f"Cost: {best_cost}, Time: {elapsed:.2f}s")
        return method_name, best_cost
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Error after {elapsed:.2f}s: {e}")
        return method_name, float('inf')


def process_test_case(test_file, input_dir, output_file):
    """Process a single test case"""
    test_name = os.path.splitext(os.path.basename(test_file))[0]
    print(f"\n{'='*60}")
    print(f"Processing {test_name}")
    print(f"{'='*60}")
    
    # Read input
    filepath = os.path.join(input_dir, test_file)
    n, customers, travel_time = read_input_from_file(filepath)
    print(f"Number of customers: {n}")
    
    # Get time limit based on problem size
    time_limit = get_time_limit(n)
    print(f"Time limit: {time_limit}s per method")
    
    results = []
    
    # Run Local Search modes (1-6)
    print("\nLocal Search Methods:")
    for mode in range(1, 7):
        method_name = f"LocalSearch_Mode{mode}"
        method_func = lambda tl, m=mode: run_local_search_mode(m, n, customers, travel_time, tl)
        result = run_method(method_name, method_func, time_limit)
        results.append(result)
    
    # Run Tabu Search modes (1-3)
    print("\nTabu Search Methods:")
    for mode in range(1, 4):
        method_name = f"TabuSearch_Mode{mode}"
        method_func = lambda tl, m=mode: run_tabu_search_mode(m, n, customers, travel_time, tl)
        result = run_method(method_name, method_func, time_limit)
        results.append(result)
    
    # Run Genetic Algorithm
    print("\nGenetic Algorithm:")
    method_name = "GeneticAlgorithm"
    method_func = lambda tl: run_ga(n, customers, travel_time, tl)
    result = run_method(method_name, method_func, time_limit)
    results.append(result)
    
    # Run Exact Algorithms (for all test cases)
    print("\nExact Algorithms:")
    
    # Backtracking
    print(f"  Running Backtracking...", end=' ', flush=True)
    result = backtracking_solve(n, customers, travel_time, time_limit)
    if result[1] == "TLE":
        print(f"TLE (nodes: {result[2]}, time: {result[3]:.2f}s)")
        results.append(("Backtracking", "TLE"))
    else:
        route, cost, nodes, elapsed = result
        print(f"Cost: {cost}, Nodes: {nodes}, Time: {elapsed:.2f}s")
        results.append(("Backtracking", cost))
    
    # Branch and Bound
    print(f"  Running Branch&Bound...", end=' ', flush=True)
    result = branch_and_bound_solve(n, customers, travel_time, time_limit)
    if result[1] == "TLE":
        print(f"TLE (nodes: {result[2]}, pruned: {result[3]}, time: {result[4]:.2f}s)")
        results.append(("BranchAndBound", "TLE"))
    else:
        route, cost, nodes, pruned, elapsed = result
        print(f"Cost: {cost}, Nodes: {nodes}, Pruned: {pruned}, Time: {elapsed:.2f}s")
        results.append(("BranchAndBound", cost))
    
    # OR-Tools
    print(f"  Running OR-Tools...", end=' ', flush=True)
    result = ortools_solve(n, customers, travel_time, time_limit)
    if isinstance(result[1], str) and ("TLE" in result[1] or "ERROR" in result[1]):
        print(f"{result[1]} (time: {result[3]:.2f}s)")
        results.append(("ORTools", result[1].split()[0]))  # Extract TLE or ERROR
    else:
        route, cost, _, elapsed = result
        print(f"Cost: {cost}, Time: {elapsed:.2f}s")
        results.append(("ORTools", cost))
    
    # Write results
    with open(output_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Test Case: {test_name}\n")
        f.write(f"{'='*60}\n")
        for method_name, cost in results:
            if cost == float('inf'):
                f.write(f"{method_name}: FAILED\n")
            elif cost == "TLE":
                f.write(f"{method_name}: TLE\n")
            elif cost == "ERROR":
                f.write(f"{method_name}: ERROR\n")
            else:
                f.write(f"{method_name}: {cost}\n")
    
    print(f"\nResults for {test_name} written to {output_file}")


def main():
    """
    Main function - runs all algorithms including exact methods for small instances
    """
    # Setup paths (relative to current directory)
    input_dir = "input"
    output_file = "result.txt"
    
    # Get all test case files (both original and generated versions)
    # Find all .txt files that start with 'N' and contain digits
    all_files = []
    if os.path.exists(input_dir):
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt') and filename.startswith('N') and any(c.isdigit() for c in filename):
                all_files.append(filename)
    
    # Sort files by size and version
    def sort_key(filename):
        # Extract number from filename (e.g., N5.txt -> 5, N100_v2.txt -> 100)
        match = re.search(r'N(\d+)', filename)
        if match:
            size = int(match.group(1))
            # Check if it has version number
            version_match = re.search(r'_v(\d+)', filename)
            version = int(version_match.group(1)) if version_match else 0
            return (size, version)
        return (0, 0)
    
    all_files.sort(key=sort_key)
    
    print(f"{'='*60}")
    print("VRPTW Optimization - Running All Methods")
    print(f"Found {len(all_files)} test cases")
    print(f"{'='*60}")
    print("\nAlgorithms to run (13 total):")
    print("  - 6 Local Search modes")
    print("  - 3 Tabu Search modes")
    print("  - 1 Genetic Algorithm")
    print("  - 3 Exact algorithms (Backtracking, Branch&Bound, OR-Tools)")
    print("\nNote: Exact algorithms may TLE for large instances (N > 15)")
    print(f"{'='*60}\n")
    
    # Clear output file
    with open(output_file, 'w') as f:
        f.write("VRPTW Optimization Results - All Algorithms\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total test cases: {len(all_files)}\n")
        f.write(f"\nAlgorithms (13 total):\n")
        f.write(f"  - Metaheuristics: 6 Local Search + 3 Tabu Search + 1 GA = 10 methods\n")
        f.write(f"  - Exact: Backtracking + Branch&Bound + OR-Tools = 3 methods\n")
        f.write(f"\nNote: Exact algorithms may show TLE for large instances (N > 15)\n\n")
    
    # Process each test case
    total_start = time.time()
    completed = 0
    
    for test_file in all_files:
        filepath = os.path.join(input_dir, test_file)
        if os.path.exists(filepath):
            try:
                process_test_case(test_file, input_dir, output_file)
                completed += 1
                
                # Print progress
                elapsed = time.time() - total_start
                avg_time = elapsed / completed if completed > 0 else 0
                remaining = len(all_files) - completed
                estimated_remaining = avg_time * remaining
                
                print(f"\n{'='*60}")
                print(f"Progress: {completed}/{len(all_files)} ({100*completed/len(all_files):.1f}%)")
                print(f"Elapsed time: {elapsed/60:.1f} minutes")
                if remaining > 0:
                    print(f"Estimated remaining: {estimated_remaining/60:.1f} minutes")
                print(f"{'='*60}")
                
            except Exception as e:
                print(f"Error processing {test_file}: {e}")
                with open(output_file, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Test Case: {test_file}\n")
                    f.write(f"ERROR: {e}\n")
        else:
            print(f"Warning: {test_file} not found, skipping...")
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"All tests completed!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
