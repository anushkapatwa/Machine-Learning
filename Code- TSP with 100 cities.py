import random  
import math  
import time  
import matplotlib.pyplot as plt  
import numpy as np


# Method to calculate the total distance of a tour
# Parameters: Randomly generated tour, randomly generated distance 
# Returns: The total distance of the given tour
def total_distance(tour, distances):
    total = 0
    for i in range(99):
        total += distances[tour[i]][tour[i + 1]]  
    total += distances[tour[-1]][tour[0]] 
    return total

# Method to create a random tour with 100 cities. 
# Parameters: None
# Returns: Random tour with 100 cities
def initial_tour():
    tour = random.sample(range(100),100)
    return tour

# Method to generate neighbors
# Parameters: Tour whose neighbor we want
# Returns: Possible neighbors of the given tour
def generate_neighbors(tour):
    neighbors = []
    for i in range(len(tour)):
        for j in range(i + 1, len(tour)):
            new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]  # Performing a swap on the tour
            neighbors.append(new_tour)
    return neighbors


# Simulated Annealing Algorithm
# Parameters: Distances list, intital temp, colling rate, minimum temperature
# Returns: Best tour, Best distance, temperature, number of iterations, best distance for each iteration, temperature at best distance found
def simulated_annealing(distances, initial_temp, cooling_rate, min_temp):
    temperatures = []
    iterations = []
    temperature_of_best_distance = [] # List to store the temperature when best distance is found
    best_distance_history = []
    current_tour = initial_tour() 
    current_distance = total_distance(current_tour, distances)  
    best_tour = current_tour[:]  
    best_distance = current_distance  
    temp = initial_temp  
    iteration = 0  

    while temp > min_temp:
        new_neighbor = random.choice(generate_neighbors(current_tour)) # New neighbor created
        neighbor_distance = total_distance(new_neighbor, distances)  
        
        # Checking to accept new neighbor or not
        if neighbor_distance < current_distance or random.random() < math.exp((current_distance - neighbor_distance) / temp):
            current_tour = new_neighbor[:]  
            current_distance = neighbor_distance  
            
            if current_distance < best_distance:
                best_tour = current_tour[:]  
                best_distance = current_distance  
                best_distance_history.append(best_distance)
                temperature_of_best_distance.append(temp)
            
       
        temperatures.append(temp)
        iterations.append(iteration)
        
        iteration = iteration + 1  
        temp = temp * cooling_rate
        
    return best_tour, best_distance, temperatures, iterations, best_distance_history, temperature_of_best_distance

# Threshold Accepting Algorithm
# Parameters: Distances list, threshold values, maximum time for each threshold
# Returns: Best tour, best distance, dictionary containing best solution for each threshold
def threshold_accepting(distances, thresholds, max_time):

    best_solutions = {}  # Dictionary to store the best solution for each threshold
    start_time = time.time()  
    iteration = 0 

    for threshold in thresholds:
        current_tour = initial_tour()  
        current_distance = total_distance(current_tour, distances)  
        best_tour = current_tour[:]  
        best_distance = current_distance 
        
        while time.time() - start_time < max_time:  
            new_neighbor = random.choice(generate_neighbors(current_tour))  
            neighbor_distance = total_distance(new_neighbor, distances)  
            
            # Accepting the new tour if it improves the distance within the threshold
            if neighbor_distance <= current_distance + threshold:
                current_tour = new_neighbor[:]  
                current_distance = neighbor_distance  
                
                if current_distance < best_distance:
                    best_tour = current_tour[:]  
                    best_distance = current_distance  
            
                
        best_solutions[threshold] = (best_tour, best_distance)  # Storing the best solution for the current threshold

    # Finding the best solution among all thresholds
    best_tour = min(best_solutions.values(), key=lambda x: x[1])[0]  # Storing the tour with the minimum distance
    best_distance = min(best_solutions.values(), key=lambda x: x[1])[1]  # Storing the minimum distance

    return best_tour, best_distance, best_solutions

# Hill Climbing Algorithm
# Parameters: Distances List, Minimum Improvement
# Returns: Best tour, Best Distance, Best distance history
def hill_climbing(distances, min_improvement):

    best_distance_history = []
    current_tour = initial_tour()  
    current_distance = total_distance(current_tour, distances)  
    best_tour = current_tour[:] 
    best_distance = current_distance  
    counter = 0  # Counter to track consecutive iterations without improvement
 

    
    while counter < 1000:  
        neighbors = generate_neighbors(current_tour)
        improved = False  # Flag to indicate an improvement
        
        for new_tour in neighbors:  
            new_distance = total_distance(new_tour, distances) 
            
            if new_distance < current_distance:  
                current_tour = new_tour[:]  
                current_distance = new_distance  
                improved = True  
                counter = 0
                break  

        
        # Incrementing the counter if no improving neighbor is found or improvements are very small
        if improved == False or best_distance - current_distance < min_improvement:
            counter = counter + 1
            

        if current_distance < best_distance:
            best_tour = current_tour[:]  
            best_distance = current_distance  
            best_distance_history.append(best_distance)
        
    return best_tour, best_distance, best_distance_history


# Main function
# Distance matrix (randomly generated)
distances = []
for i in range(100):
    row = []
    for j in range(100):
        row.append(random.randint(1, 100)) 
    distances.append(row)

initial_temp = 10000  # Initial temperature for simulated annealing
cooling_rate =  0.89 # Cooling rate for simulated annealing
min_temp = 0.01  # Minimum temperature for simulated annealing
thresholds = [800, 600, 400, 300, 200, 100]  # Threshold values for threshold accepting
max_time = 800  # Maximum time for threshold accepting in seconds
min_improvement = 0.0001  # Minimum improvement for hill climbing

# Running algorithms

# Simulated annealing
sa_best_tour, sa_best_distance, sa_temperatures, sa_iterations, sa_best_distance_history, sa_temperature_of_best_distance = simulated_annealing(distances, initial_temp, cooling_rate, min_temp)

# Threshold Accepting
ta_best_tour, ta_best_distance, ta_best_solutions = threshold_accepting(distances, thresholds, max_time)

# Hill Climbing
hc_best_tour, hc_best_distance, hc_best_distance_history = hill_climbing(distances, min_improvement)


# Plotting

# Simulated Annealing

plt.figure(figsize=(15, 5))

# Temp vs iterations graph
plt.subplot(1, 2, 1)
plt.plot(sa_iterations, sa_temperatures)
plt.title('Simulated Annealing - Temperature vs Iterations')
plt.xlabel('Iteration')
plt.ylabel('Temperature')

# Temp vs best distance graph
plt.subplot(1, 2, 2)
plt.plot(sa_temperature_of_best_distance, sa_best_distance_history, marker = 'o')
plt.title('Simulated Annealing - Temperature vs Best Distance')
plt.xlabel('Temperature')
plt.ylabel('Best Distance')

plt.show()


# Threshold Accepting
# Threshold vs best distance graph

# Extracting the best distance for each threshold
TA_BEST_DISTANCES = []
for i in ta_best_solutions.values():
    TA_BEST_DISTANCES.append(i[1])
    
plt.figure(figsize=(8, 5))
plt.plot(thresholds, TA_BEST_DISTANCES, marker='o')
plt.title('Threshold Accepting - Threshold vs Best Distance')
plt.xlabel('Threshold')
plt.ylabel('Best Distance')
plt.show()

# Hill Climbing
# Best distance vs iterations graph
plt.figure(figsize=(8, 5))
plt.plot(range(len(hc_best_distance_history)), hc_best_distance_history)
plt.title('Hill Climbing - Best Distance vs Best Distances Count')
plt.xlabel('Best Distances count')
plt.ylabel('Best Distance')
plt.show()

# Best tour and best distances
print("Simulated Annealing:")
print("Best Tour:", sa_best_tour)
print("Best Distance:", sa_best_distance)
print()
print("Threshold Accepting:")
print("Best Tour:", ta_best_tour)
print("Best Distance:", ta_best_distance)
print()
print("Hill Climbing:")
print("Best Tour:", hc_best_tour)
print("Best Distance:", hc_best_distance)
