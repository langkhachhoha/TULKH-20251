import time
import heapq

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

def branch_and_bound_solve(n, customers, travel_time, time_limit=60):
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
    
    best_route, best_cost = branch_and_bound_solve(n, customers, travel_time, time_limit)
    
    return n, best_route

if __name__ == "__main__":
    n, route = solve()
    print(n)
    print(' '.join(map(str, route)))
