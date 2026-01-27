import random
import sys
import copy

MODE = 1  

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
    new_route = route[:]
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def insert_move(route, i, j):
    new_route = route[:]
    element = new_route.pop(i)
    new_route.insert(j, element)
    return new_route

def reverse_segment(route, i, j):
    new_route = route[:]
    if i > j:
        i, j = j, i
    new_route[i:j+1] = reversed(new_route[i:j+1])
    return new_route

def get_neighborhood_candidates(route, n, max_candidates=None):
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            candidates.append(('swap', i, j))
    if max_candidates and len(candidates) > max_candidates:
        random.shuffle(candidates)
        candidates = candidates[:max_candidates]
    return candidates

def tabu_search_mode1(route, customers, travel_time, max_iter=2000):
    best_route = route[:]
    best_cost = calculate_total_time(best_route, customers, travel_time)
    current_route = route[:]
    current_cost = best_cost
    
    tabu_list = []
    tabu_tenure = 7
    frequency = {}
    
    no_improve = 0
    phase = 'intensification'  
    
    n = len(current_route)
    
    for iteration in range(max_iter):
        if no_improve > 50:
            tabu_tenure = min(15, tabu_tenure + 1)
        elif no_improve > 100:
            phase = 'diversification'
            tabu_tenure = 3
        
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None
        
        candidates = get_neighborhood_candidates(current_route, n, max_candidates=n*3)
        
        for move_type, i, j in candidates:
            move = (i, j)
            
            if move in tabu_list:
                continue
            
            neighbor = two_opt_swap(current_route, i, j)
            neighbor_cost = calculate_total_time(neighbor, customers, travel_time)
            
            if phase == 'diversification':
                penalty = frequency.get(move, 0) * 0.5
                neighbor_cost += penalty
            
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost
                best_move = move
        
        if best_neighbor is None:
            break
        
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



def solve():
    n, customers, travel_time = read_input()
    
    best_overall_route = None
    best_overall_cost = float('inf')
    
    strategies = ['nearest', 'earliest', 'random']
    
    for strategy in strategies[:2]:  
        initial_route = generate_initial_solution(n, customers, travel_time, strategy=strategy)
        
        best_route, best_cost = tabu_search_mode1(initial_route, customers, travel_time)
        
        if best_cost < best_overall_cost:
            best_overall_route = best_route
            best_overall_cost = best_cost
    
    return n, best_overall_route

if __name__ == "__main__":
    n, route = solve()
    print(n)
    print(' '.join(map(str, route)))

