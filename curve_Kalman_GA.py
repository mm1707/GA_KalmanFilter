from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import random

class Individual:
    def __init__(self, R, Q):
        self.R = R
        self.Q = Q
    def mutate(self): # Mutacja niejednorodna
        random_matrix = np.random.uniform(-0.05, 0.05, self.R.shape)
        self.R += random_matrix
        random_matrix = np.random.uniform(-0.05, 0.05, self.Q.shape)
        self.Q += random_matrix

class KalmanFilter:
    def __init__(self, F, G, H, Q, R, x0, P0):
        self.F = F  # State transition model
        self.G = G  # Control matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state
        self.P = P0  # Initial covariance
    def predict(self, u):
        if u is None:
            x = self.F @ self.x
        # Calculate the uncertainty `P` of state vector.
        P = self.F @ self.P @ self.F.T + self.Q
        # Update self.x and self.P.
        self.x = x
        self.P = P
        return self.x

    def update(self, z):
        # Calculate Kalman gain `K`.
        K = (self.P @ self.H.T @
             np.linalg.inv(self.H @ self.P @ self.H.T + self.R))
        # Calculate the updated state vector `x`.
        x = self.x + K @ (np.asarray(z) - self.H @ self.x)
        # Calculate the updated uncertainty `P` for the state vector.
        T = np.eye(K.shape[0]) - K @ self.H
        P = T @ self.P @ T.T + K @ self.R @ K.T
        # Update self.K, self.x and self.P.
        self.K = K
        self.x = x
        self.P = P
        return self.x


def fitness(true, estimated):
    return np.mean(np.sqrt((estimated-true)**2))

def elite(population_fit, number_of_individuals):
    return np.argsort(population_fit)[:number_of_individuals]

def selection(population_fit, num_select): # Stochastic universal sampling
    max_fitness = np.max(population_fit) + 1  # To avoid division by zero
    inverted_fitness = max_fitness - population_fit

    fitness_sum = np.sum(inverted_fitness)
    probabilities = inverted_fitness / fitness_sum

    cumulative_sum = np.cumsum(probabilities)

    start_point = np.random.uniform(0, 1 / num_select)

    selected_indices = []
    for i in range(num_select):
        pick = start_point + i / num_select
        selected_indices.append(np.searchsorted(cumulative_sum, pick))

    return selected_indices

def crossover(parent_a, parent_b): # Rekombinacja calkowita arytmetyczna
    return Individual((parent_a.R+parent_b.R)/2, (parent_a.Q+parent_b.Q)/2)

t = np.arange(35)
x_true = [-400 + 25 * i if i < 16 else
          300 * np.sin((25 * i - 400) / 300) for i in t]
y_true = [300 if i < 16 else
          300 * np.cos((25 * i - 400) / 300) for i in t]
# Measurements
xs = x_true+np.random.normal(0, 5, len(x_true))
ys = y_true+np.random.normal(0, 5, len(y_true))

## Kalman Setup
F = np.array([[1, 1, 0.5, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0.5],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]])
P0 = np.diag([1100] * 6)

R = np.array([[3 ** 2, 0],
     [0, 3 ** 2]])
Q = np.array((F @
     [[0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0.2 ** 2, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0.2 ** 2]]
     @ F.T))
G = None
x0 = [0, 0, 0, 0, 0, 0]



# Genetic algorithm
population_size = 25
children = 5
generations = 40
mutation_rate = 0.1

population = []
population_fitness = np.zeros(population_size)

all_fitness = []
best_individual_fitness = []

# Create initial population
for i in range(population_size):
    R = np.random.uniform(-5, 5, R.shape)
    Q = np.random.uniform(-5, 5, Q.shape)
    population.append(Individual(R, Q))

# Calculate fitness
for i in range(population_size):
    zs = zip(xs, ys)
    kf = KalmanFilter(F, G, H, population[i].Q, population[i].R, x0, P0)

    estimated_positions = []

    for z in zs:
        kf.predict(u=None)
        kf.update(z)
        estimated_positions.append(kf.x)

    x_estimated = [j[0] for j in estimated_positions]
    y_estimated = [j[3] for j in estimated_positions]

    population_fitness[i] = fitness(np.array(x_true), np.array(x_estimated))
    population_fitness[i] += fitness(np.array(y_true), np.array(y_estimated))

# Initial solution
top = population[0]
kf = KalmanFilter(F, G, H, top.Q, top.R, x0, P0)
estimated_positions = []
zs = zip(xs, ys)
for z in zs:
    kf.predict(u=None)
    kf.update(z)
    estimated_positions.append(kf.x)

print("Initial Measurement Noise Covariance Matrix R: ")
print(top.R)
print("Initial Process Noise Covariance Matrix Q: ")
print(top.Q)
print("Initial Fitness: ")
print(fitness(np.array(x_true), np.array([i[0] for i in estimated_positions])))

fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(111)
ax.plot(x_true, y_true, 'gd-', label='Model')
ax.plot(xs, ys, 'bs-', label='Measurements')
ax.plot([i[0] for i in estimated_positions],
        [i[3] for i in estimated_positions],
        'ro-', label='Estimates')
ax.set_xlabel('$x$ (m)')
ax.set_ylabel('$y$ (m)')
ax.set_xlim(-400, 320)
ax.set_ylim(0, 320)
ax.legend()
ax.grid()
fig.suptitle('Kalman Filter for Particle Tracking - First Iteration')
plt.show()


# Main loop of the evolutionary algorithm
for i in range(generations):
    parents_indices = selection(population_fitness, children*2)
    for j in range(0, 2 * children, 2):
        child = crossover(population[parents_indices[j]], population[parents_indices[j+1]])
        child.mutate()
        population = np.append(population, child)

    population_fitness = np.zeros(population_size+children)
    for h in range(population.size):
        kf = KalmanFilter(F, G, H, population[h].Q, population[h].R, x0, P0)
        estimated_positions = []
        zs = zip(xs, ys)
        for z in zs:
            kf.predict(u=None)
            kf.update(z)
            estimated_positions.append(kf.x)

        x_estimated = [j[0] for j in estimated_positions]
        y_estimated = [j[3] for j in estimated_positions]

        population_fitness[h] = fitness(np.array(x_true), np.array(x_estimated))
        population_fitness[h] += fitness(np.array(y_true), np.array(y_estimated))

    # elityzm
    elite_indices = elite(population_fitness, population_size)
    population = population[elite_indices]
    population_fitness = population_fitness[elite_indices]


    all_fitness.append(np.mean(population_fitness))
    best_individual_fitness.append(np.min(population_fitness))

# Top individual
top = population[elite(population_fitness, 1)]
kf = KalmanFilter(F, G, H, top[0].Q, top[0].R, x0, P0)
estimated_positions = []
zs = zip(xs, ys)
for z in zs:
    kf.predict(u=None)
    kf.update(z)
    estimated_positions.append(kf.x)

print("Optimized Measurement Noise Covariance Matrix R: ")
print(top[0].R)
print("Optimized Process Noise Covariance Matrix Q: ")
print(top[0].Q)
print("Fitness: ")
print(fitness(np.array(x_true), np.array([i[0] for i in estimated_positions])))

fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(111)
ax.plot(x_true, y_true, 'gd-', label='Model')
ax.plot(xs, ys, 'bs-', label='Measurements')
ax.plot([i[0] for i in estimated_positions],
        [i[3] for i in estimated_positions],
        'ro-', label='Top Estimates')
ax.set_xlabel('$x$ (m)')
ax.set_ylabel('$y$ (m)')
ax.legend()
ax.grid()
ax.set_aspect('equal')
fig.suptitle('Kalman Filter for Particle Tracking - Last Iteration')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(all_fitness, label='mean fitness of population', color='green', linestyle='-')
plt.title('Fitness of Kalman Filter for Particle Tracking with GA')
plt.xlabel('Generation')
plt.ylabel('Mean fitness')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(best_individual_fitness, label='fitness of the best individual', color='blue', linestyle='-')
plt.title('Fitness of Kalman Filter for Particle Tracking with GA')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.show()
