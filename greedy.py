import random

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

def greedy_nearest_neighbor(n, customers, travel_time, t0=0):
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
    
    return route

def greedy_earliest_deadline(n, customers, travel_time, t0=0):
    route = sorted(range(1, n + 1), key=lambda x: customers[x-1]['earliest'])
    return route

def greedy_latest_deadline(n, customers, travel_time, t0=0):
    route = sorted(range(1, n + 1), key=lambda x: customers[x-1]['latest'])
    return route

def greedy_urgency_ratio(n, customers, travel_time, t0=0):
    route = []
    visited = set()
    current_time = t0
    current_pos = 0
    
    while len(route) < n:
        best_next = -1
        best_ratio = float('inf')
        
        for i in range(1, n + 1):
            if i in visited:
                continue
            
            travel = travel_time[current_pos][i]
            arrival_time = current_time + travel
            customer = customers[i - 1]
            
            if arrival_time <= customer['latest']:
                time_window = customer['latest'] - customer['earliest']
                urgency_ratio = (time_window + 1) / (travel + 1)  
                
                if urgency_ratio < best_ratio:
                    best_ratio = urgency_ratio
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
        travel = travel_time[current_pos][best_next]
        arrival_time = current_time + travel
        start_service = max(arrival_time, customer['earliest'])
        current_time = start_service + customer['duration']
        current_pos = best_next
    
    return route

def solve():
    n, customers, travel_time = read_input()
    
    best_overall_route = None
    best_overall_cost = float('inf')
    
    strategies = [
        ('nearest', greedy_nearest_neighbor),
        ('earliest', greedy_earliest_deadline),
        ('latest', greedy_latest_deadline),
        ('urgency', greedy_urgency_ratio)
    ]
    
    for strategy_name, strategy_func in strategies:
        route = strategy_func(n, customers, travel_time)
        
        if is_feasible(route, customers, travel_time):
            cost = calculate_total_time(route, customers, travel_time)
            if cost < best_overall_cost:
                best_overall_route = route
                best_overall_cost = cost
    
    if best_overall_route is None:
        best_overall_route = greedy_nearest_neighbor(n, customers, travel_time)
    
    return n, best_overall_route

if __name__ == "__main__":
    n, route = solve()
    print(n)
    print(' '.join(map(str, route)))
