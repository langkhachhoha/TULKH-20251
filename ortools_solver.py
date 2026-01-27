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

def ortools_solve(n, customers, travel_time, time_limit=60):
    try:
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
    except ImportError:
        print("ERROR: OR-Tools not installed", file=sys.stderr)
        return list(range(1, n + 1)), float('inf')
    
    start_time = time.time()
    
    manager = pywrapcp.RoutingIndexManager(n + 1, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if to_node == 0:  
            return 0
        return travel_time[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel = travel_time[from_node][to_node]
        service_time = customers[from_node - 1]['duration'] if from_node > 0 else 0
        return travel + service_time
    
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    
    routing.AddDimension(
        time_callback_index,
        100000,  
        100000,  
        False,
        'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')
    
    for customer_idx in range(1, n + 1):
        index = manager.NodeToIndex(customer_idx)
        customer = customers[customer_idx - 1]
        time_dimension.CumulVar(index).SetRange(customer['earliest'], customer['latest'])
    
    depot_idx = manager.NodeToIndex(0)
    time_dimension.CumulVar(depot_idx).SetRange(0, 0)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = int(time_limit)
    search_parameters.log_search = False
    
    solution = routing.SolveWithParameters(search_parameters)
    
    elapsed = time.time() - start_time
    
    if solution:
        route = []
        index = routing.Start(0)
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                route.append(node)
            index = solution.Value(routing.NextVar(index))
        
        cost = int(solution.ObjectiveValue())
        return route, cost
    else:
        return list(range(1, n + 1)), float('inf')

def solve():
    n, customers, travel_time = read_input()
    
    if n <= 10:
        time_limit = 60
    elif n <= 100:
        time_limit = 60
    elif n <= 500:
        time_limit = 120
    else:
        time_limit = 180
    
    best_route, best_cost = ortools_solve(n, customers, travel_time, time_limit)
    
    return n, best_route

if __name__ == "__main__":
    n, route = solve()
    print(n)
    print(' '.join(map(str, route)))
