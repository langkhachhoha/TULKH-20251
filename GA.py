import random
import sys
from typing import List, Tuple

class GA_VRPTW:
    def __init__(self, n: int, time_windows: List[Tuple[int, int]], 
                 service_times: List[int], travel_matrix: List[List[int]],
                 pop_size: int = 100, max_gen: int = 500, 
                 crossover_rate: float = 0.9, mutation_rate: float = 0.2):
        self.n = n
        self.e = [0] + [tw[0] for tw in time_windows]  
        self.l = [0] + [tw[1] for tw in time_windows]  
        self.d = [0] + service_times                   
        self.t = travel_matrix
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = []
        self.best_solution = None
        self.best_cost = float('inf')
    
    def calculate_cost(self, chromosome: List[int]) -> Tuple[int, int]:
        total_travel = 0
        current_time = 0
        current_pos = 0  
        penalty = 0
        
        for customer in chromosome:
            travel_time = self.t[current_pos][customer]
            total_travel += travel_time
            arrival_time = current_time + travel_time
            
            # Chờ nếu đến sớm
            if arrival_time < self.e[customer]:
                current_time = self.e[customer]
            else:
                current_time = arrival_time
            
            # Penalty nếu đến muộn
            if current_time > self.l[customer]:
                penalty += (current_time - self.l[customer]) * 10000
            
            current_time += self.d[customer]
            current_pos = customer
        
        return total_travel, penalty
    
    def fitness(self, chromosome: List[int]) -> float:
        travel, penalty = self.calculate_cost(chromosome)
        return -(travel + penalty)
    
    def is_feasible(self, chromosome: List[int]) -> bool:
        current_time = 0
        current_pos = 0
        for customer in chromosome:
            arrival_time = current_time + self.t[current_pos][customer]
            if arrival_time < self.e[customer]:
                current_time = self.e[customer]
            else:
                current_time = arrival_time
            if current_time > self.l[customer]:
                return False
            current_time += self.d[customer]
            current_pos = customer
        return True
    
    def initialize_population(self):
        self.population = []
        
        for _ in range(self.pop_size // 5):
            sol = self.greedy_solution()
            self.population.append(sol)
        
        while len(self.population) < self.pop_size:
            sol = list(range(1, self.n + 1))
            random.shuffle(sol)
            self.population.append(sol)
    
    def greedy_solution(self) -> List[int]:
        customers = list(range(1, self.n + 1))
        customers.sort(key=lambda x: (self.e[x], self.l[x]))
        return customers
    
    def tournament_selection(self, k: int = 3) -> List[int]:
        candidates = random.sample(self.population, k)
        return max(candidates, key=self.fitness)
    
    def crossover_OX(self, parent1: List[int], parent2: List[int]) -> List[int]:
        size = len(parent1)
        
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size - 1)
        
        child = [-1] * size
        child[start:end+1] = parent1[start:end+1]
        
        copied = set(child[start:end+1])
        remaining = [x for x in parent2 if x not in copied]
        
        idx = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = remaining[idx]
                idx += 1
        
        return child
    
    def mutation_swap(self, chromosome: List[int]) -> List[int]:
        mutated = chromosome.copy()
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def mutation_inversion(self, chromosome: List[int]) -> List[int]:
        mutated = chromosome.copy()
        i, j = sorted(random.sample(range(len(mutated)), 2))
        mutated[i:j+1] = mutated[i:j+1][::-1]
        return mutated
    
    def local_search_2opt(self, chromosome: List[int]) -> List[int]:
        best = chromosome.copy()
        best_cost = self.calculate_cost(best)[0] + self.calculate_cost(best)[1]
        improved = True
        
        while improved:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 2, len(best)):
                    new_sol = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    new_cost = self.calculate_cost(new_sol)[0] + self.calculate_cost(new_sol)[1]
                    
                    if new_cost < best_cost:
                        best = new_sol
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
        
        return best
    
    def evolve(self):
        self.initialize_population()
        
        for sol in self.population:
            travel, penalty = self.calculate_cost(sol)
            cost = travel + penalty
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = sol.copy()
        
        for gen in range(self.max_gen):
            new_population = []
            
            sorted_pop = sorted(self.population, key=self.fitness, reverse=True)
            elite_size = max(2, self.pop_size // 20)
            new_population.extend([sol.copy() for sol in sorted_pop[:elite_size]])
            
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                if random.random() < self.crossover_rate:
                    child = self.crossover_OX(parent1, parent2)
                else:
                    child = parent1.copy() if random.random() < 0.5 else parent2.copy()
                
                if random.random() < self.mutation_rate:
                    if random.random() < 0.5:
                        child = self.mutation_swap(child)
                    else:
                        child = self.mutation_inversion(child)
                
                new_population.append(child)
            
            self.population = new_population
            
            for sol in self.population:
                travel, penalty = self.calculate_cost(sol)
                cost = travel + penalty
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = sol.copy()
            
            if gen % 50 == 0 and self.best_solution:
                improved = self.local_search_2opt(self.best_solution)
                travel, penalty = self.calculate_cost(improved)
                cost = travel + penalty
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = improved.copy()
        
        return self.best_solution, self.best_cost


def read_input():
    n = int(input().strip())
    
    time_windows = []
    service_times = []
    for _ in range(n):
        line = input().strip().split()
        e_i = int(line[0])
        l_i = int(line[1])
        d_i = int(line[2])
        time_windows.append((e_i, l_i))
        service_times.append(d_i)
    
    travel_matrix = []
    for _ in range(n + 1):
        line = input().strip().split()
        row = [int(x) for x in line]
        travel_matrix.append(row)
    
    return n, time_windows, service_times, travel_matrix


def main():
    n, time_windows, service_times, travel_matrix = read_input()
    
    ga = GA_VRPTW(
        n=n,
        time_windows=time_windows,
        service_times=service_times,
        travel_matrix=travel_matrix,
        pop_size=100,
        max_gen=100,
        crossover_rate=0.9,
        mutation_rate=0.2
    )
    
    best_solution, best_cost = ga.evolve()
    
    print(n)
    print(' '.join(map(str, best_solution)))


if __name__ == "__main__":
    random.seed(42)
    main()


