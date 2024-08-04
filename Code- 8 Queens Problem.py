import random
import time
import matplotlib.pyplot as plt
import numpy as np

class EightQueensSolver:
    
    
# Construtor for the solver, containing parameters used to solve the problem
# Parameters: Solution Space, mutation probability, number of offsprings, termination condition
# Returns: None
    def __init__(self, population_size=100, mutation_prob=0.8, offspring_count=2, termination_limit=10000):
        self.population_size = population_size      # Number of potential solutions
        self.mutation_prob = mutation_prob          # Probability of mutation
        self.offspring_count = offspring_count      # Number of offspring created in each step
        self.termination_limit = termination_limit  # Total number of steps before termination
        self.best_fitness_history = []              # List to store the fitness history over iterations
        self.solution = None                        # Store the final solution
    

# Method to initialize a solution space with specified attempts(population size)
# Parameters: None
# Returns: Solution Space with specified number of attempts
    def initialize_population(self):
        population = []
        for i in range(self.population_size):
            population.append(random.sample(range(8),8))
        return population


# Method to randomly select 5 parents
# Parameters: Solution Space
# Returns: 5 Random attemps from solution space(Parents)
    def parents_selection(self, population):
        return random.sample(population, 5)
    

# Method to perform cut-cross-fill crossover to generate offsprings
# Parameters: First two parents
# Returns: offsprings 
    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, 6)
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        return offspring1, offspring2
    
    
# Method to perform swap mutation on the offsprings with certain probability
# Parameters: Offspring
# Returns: Mutated offspring
    def mutation(self, offspring):
        if random.random() < self.mutation_prob:
            index1, index2 = random.sample(range(8), 2)
            temp = offspring[index2]
            offspring[index2] = offspring[index1]
            offspring[index1] = temp
        return offspring


# Method to update the solution space with the offsprings
# Parameters: Solution space and the offsprings
# Returns: The updated and sorted solution space
    def update_population(self, population, offspring):
        population.sort(key=lambda x: self.fitness_score(x), reverse=True)
        for i in range(len(offspring)):
            population[-(i+1)] = offspring[i]
        return population
    

# Method to calculate the fitness score for each attempt in the solution space
# Parameter: An attempt from the solution space
# Returns: The fitness score of the attempt
    def fitness_score(self, attempt):
        penalty = 0
        for queen_index in range(8):
            penalty += self.penalty(attempt, queen_index)
        return 1 / (penalty + 1)
    

# Method to calculate the penalty for each queen
# Parameters: The attempt from which the queen is taken and the index of the queen
# Returns: The penalty for the queen
    def penalty(self, attempt, queen_index):
        penalty = 0
        for i in range(8):
            if i != queen_index:
                if attempt[i] == attempt[queen_index] or abs(i - queen_index) == abs(attempt[i] - attempt[queen_index]):
                    penalty += 1
        return penalty


# Method to print the board with the queens placed
# Parameters: Attempt with the best fitness score
# Returns: Prints the board
    def print_board(self, individual):
        for i in range(8):
            row = []
            for j in range(8):
                if individual[i] == j:
                    row.append(".Q")
                else:
                    row.append(". ")
            print("".join(row))
        print("\n" + "-" * 17)

        
# Method to execute the evolutionary algorithm to solve the Eight Queens problem.
# Returns: A tuple containing the final solution and the time taken for the algorithm to execute.    
    def evolutionary_algorithm(self):
        population = self.initialize_population()
        iteration = 0
        start_time = time.time()
        
        while iteration < self.termination_limit:
            parents = self.parents_selection(population)
            offspring = []
            for i in range(self.offspring_count):
                offspring1, offspring2 = self.crossover(parents[0], parents[1])
                offspring1 = self.mutation(offspring1)
                offspring2 = self.mutation(offspring2)
                offspring.append(offspring1)
                offspring.append(offspring2)
            population = self.update_population(population, offspring)
            iteration += 1
            best_solution = population[0]      # Assigning the best solution from the population (solution space)
            self.best_fitness_history.append(self.fitness_score(best_solution)) # Appending the fitness value of the best solution
            self.solution = best_solution  # Store the final solution for a particular iteration
        
        end_time = time.time()
        return self.solution, end_time - start_time
    

# Method to call the evolutionary algorithm multiple times. 
# Then calculating the mean of the best fitness history for each trial
# Parameters: Population_size, mutation_prob, offspring_count, termination_limit, trials
# Returns:    Convergence graph, Average time taken, final solution
def solve(population_size, mutation_prob, offspring_count, termination_limit, trials):
    total_best_fitness_history = []  # Store the fitness history of each run in a nested list[2-d arrays]
    total_time = 0  # Store total time for all runs
    
    for i in range(trials):
        solver = EightQueensSolver(population_size, mutation_prob, offspring_count, termination_limit)
        solution, run_time = solver.evolutionary_algorithm()  # Get final solution and time
        total_best_fitness_history.append(solver.best_fitness_history)  # Appending the fitness history from each run
        total_time += run_time  # Add time for each run to the previous runs
    
    # Calculating mean fitness score for each iteration across all trials
    mean_fitness = np.mean(total_best_fitness_history, axis=0)
    
    # Plot line 
    plt.plot(range(termination_limit), mean_fitness, color='blue')
    plt.title("Average Fitness Over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Average Fitness")
    plt.show()
    
    # Printing final board for the best solution from the last run
    print("Best Solution:")
    solver.print_board(solution)
    print("Positions:", solution)
    
    # Printing average time taken
    print("Average time taken for each trial:", total_time / trials, "seconds")



# Parameters for the solver
population_size = 100
mutation_prob = 0.8
offspring_count = 2
termination_limit = 10000
trials = 100

solve(population_size, mutation_prob, offspring_count, termination_limit, trials)
