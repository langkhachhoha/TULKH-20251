import random
import sys
import copy
import math

MODE = 2  

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

def is_feasible(route, customers, travel_time, t0=0):
    return calculate_total_time(route, customers, travel_time, t0) != float('inf')

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
    
    elif strategy == 'latest':
        route = sorted(range(1, n + 1), key=lambda x: customers[x-1]['latest'])
    
    else:
        route = list(range(1, n + 1))
        random.shuffle(route)
    
    return route

def two_opt_swap(route, i, j):
    new_route = route[:]
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def three_opt_swap(route, i, j, k):
    new_route = route[:]
    new_route[i], new_route[j], new_route[k] = new_route[k], new_route[i], new_route[j]
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

def random_swap(route):
    new_route = route[:]
    i, j = random.sample(range(len(route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def local_search_mode2(route, customers, travel_time, max_iter=10000):
    best_route = route[:]
    best_cost = calculate_total_time(best_route, customers, travel_time)
    
    n = len(route)
    neighborhoods = ['swap', 'insert', 'reverse']
    
    iterations = 0
    while iterations < max_iter:
        improved = False
        iterations += 1
        
        for neighborhood in neighborhoods:
            if neighborhood == 'swap':
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



def solve():
    n, customers, travel_time = read_input()
    
    best_overall_route = None
    best_overall_cost = float('inf')
    
    strategies = ['nearest', 'earliest', 'latest']
    
    for strategy in strategies:
        initial_route = generate_initial_solution(n, customers, travel_time, strategy=strategy)
        

        best_route, best_cost = local_search_mode2(initial_route, customers, travel_time, max_iter=10000)
        
        if best_cost < best_overall_cost:
            best_overall_route = best_route
            best_overall_cost = best_cost
    
    return n, best_overall_route

if __name__ == "__main__":
    n, route = solve()
    print(n)
    print(' '.join(map(str, route)))

