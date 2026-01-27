import time

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

def backtracking_solve(n, customers, travel_time, time_limit=60):
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
        # Return a simple greedy solution as fallback
        best_route = list(range(1, n + 1))
    
    return best_route, best_cost

def solve():
    n, customers, travel_time = read_input()
    
    # Set time limit based on problem size
    if n <= 10:
        time_limit = 60
    elif n <= 100:
        time_limit = 60
    elif n <= 500:
        time_limit = 120
    else:
        time_limit = 180
    
    best_route, best_cost = backtracking_solve(n, customers, travel_time, time_limit)
    
    return n, best_route

if __name__ == "__main__":
    n, route = solve()
    print(n)
    print(' '.join(map(str, route)))
