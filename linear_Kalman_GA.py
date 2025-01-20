import numpy as np
import matplotlib.pyplot as plt
import random

class Individual:
    def __init__(self, R, Q):
        self.R = R
        self.Q = Q
    # def mutate(self, mutation_rate): # jednorodna
    #     if random.random() < mutation_rate:
    #         self.R = random.uniform(-5, 5)
    #     if random.random() < mutation_rate:
    #         self.Q = np.random.uniform(-5, 5, self.Q.shape)
    def mutate(self): # Mutacja niejednorodna
        self.R += random.uniform(-0.05, 0.05)
        random_matrix = np.random.uniform(-0.05, 0.05, self.Q.shape)
        self.Q += random_matrix

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, x0, P0):
        self.A = A  # State transition model
        self.B = B  # Control matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state
        self.P = P0  # Initial covariance
    def predict(self, u):
        # Predict the state and state covariance
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x
    def update(self, z):
        # Compute the Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # Update the state estimate and covariance matrix
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        return self.x


def fitness(true, estimated): # Różnica między modelem, a danymi
    return np.mean(np.sqrt((estimated-true)**2))

def elite(population_fit, number_of_individuals):
    return np.argsort(population_fit)[:number_of_individuals]

def worst(population_fit, number_of_individuals):
    return np.argsort(population_fit)[number_of_individuals:][::-1]

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


# Set the simulation parameters
num_steps = 50
true_position = np.zeros(num_steps)
true_velocity = np.ones(num_steps) * 0.5  # Constant velocity
noisy_measurements = np.zeros(num_steps)

# Generate true positions and noisy measurements
for i in range(1, num_steps):
    true_position[i] = true_position[i - 1] + true_velocity[i - 1]  # Position = previous_position + velocity
    noisy_measurements[i] = true_position[i] + np.random.normal(0, 1)  # Adding Gaussian noise

# Initialize Kalman Filter parameters
x0 = np.array([0, 0])  # Initial position and velocity (zero)
P0 = np.eye(2) * 1000  # Initial large uncertainty
H = np.array([[1, 0]])
A = np.array([[1, 1], [0, 1]])
B = np.array([[0.5], [1]])

Q = np.array([[1, 0], [0, 1]])  # Process noise
R = 1  # Measurement noise (standard deviation)

# Genetic algorithm
population_size = 25
children = 10
generations = 40
mutation_rate = 0.1

population = []
population_fitness = np.zeros(population_size)

all_fitness = []
best_individual_fitness = []

# Create initial population
for i in range(population_size):
    R = random.uniform(-5, 5)
    Q = np.random.uniform(-5, 5, Q.shape)
    population.append(Individual(R, Q))

# Calculate fitness
for i in range(population_size):
    kf = KalmanFilter(A, B, H, population[i].Q, population[i].R, x0, P0)
    estimated_positions = []
    estimated_velocities = []

    u = np.array([[1]])
    # Apply Kalman Filter for each step
    for j in range(num_steps):
        kf.predict(u)  # Predict the next state
        kf.update(noisy_measurements[j])  # Update with the noisy measurement

        # Store the estimates
        estimated_positions.append(kf.x[0, 0])
        estimated_velocities.append(kf.x[1, 1])

    population_fitness[i] = fitness(np.array(true_position), np.array(estimated_positions))

# Initial solution
ini = worst(population_fitness, 1)
ini=population[ini[0]]
print("Initial Measurement Noise Covariance Matrix R: ")
print(ini.R)
print("Initial Process Noise Covariance Matrix Q: ")
print(ini.Q)
kf = KalmanFilter(A, B, H, ini.Q, ini.R, x0, P0)
estimated_positions = np.zeros(num_steps)
estimated_velocities = np.zeros(num_steps)

u = np.array([[1]])
# Apply Kalman Filter for each step
for i in range(num_steps):
    kf.predict(u)  # Predict the next state
    kf.update(noisy_measurements[i])  # Update with the noisy measurement

    # Store the estimates
    estimated_positions[i] = kf.x[0, 0]
    estimated_velocities[i] = kf.x[1, 1]

print("Initial Fitness: ")
print(fitness(true_position, estimated_positions))
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(true_position, label='Model', color='green', linestyle='--')

# Noisy measurements
plt.scatter(range(num_steps), noisy_measurements, label='Measurements', color='red', marker='x')

# Kalman filter estimates
plt.plot(estimated_positions, label='Estimates', color='blue', linestyle='-.')

plt.title('Kalman Filter for Particle Tracking - First Iteration')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.ylim(0, 30)
plt.legend()
plt.grid(True)
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
        kf = KalmanFilter(A, B, H, population[h].Q, population[h].R, x0, P0)
        estimated_positions = []
        estimated_velocities = []

        u = np.array([[1]])
        # Apply Kalman Filter for each step
        for k in range(num_steps):
            kf.predict(u)  # Predict the next state
            kf.update(noisy_measurements[k])  # Update with the noisy measurement

            # Store the estimates
            estimated_positions.append(kf.x[0, 0])
            estimated_velocities.append(kf.x[1, 1])

        population_fitness[h] = fitness(np.array(true_position), np.array(estimated_positions))

    # elityzm
    elite_indices = elite(population_fitness, population_size)
    population = population[elite_indices]
    population_fitness = population_fitness[elite_indices]

    all_fitness.append(np.mean(population_fitness))
    best_individual_fitness.append(np.min(population_fitness))

# Top individual
top = population[elite(population_fitness, 1)]

kf = KalmanFilter(A, B, H, top[0].Q, top[0].R, x0, P0)
estimated_positions = np.zeros(num_steps)
estimated_velocities = np.zeros(num_steps)

u = np.array([[1]])
# Apply Kalman Filter for each step
for i in range(num_steps):
    kf.predict(u)  # Predict the next state
    kf.update(noisy_measurements[i])  # Update with the noisy measurement

    # Store the estimates
    estimated_positions[i] = kf.x[0, 0]
    estimated_velocities[i] = kf.x[1, 1]

print("Optimized Measurement Noise Covariance Matrix R: ")
print(top[0].R)
print("Optimized Process Noise Covariance Matrix Q: ")
print(top[0].Q)
print("Fitness: ")
print(fitness(true_position, estimated_positions))
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(true_position, label='Model', color='green', linestyle='--')

# Noisy measurements
plt.scatter(range(num_steps), noisy_measurements, label='Measurements', color='red', marker='x')

# Kalman filter estimates
plt.plot(estimated_positions, label='Estimates', color='blue', linestyle='-.')

plt.title('Kalman Filter for Particle Tracking - Last Iteration')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.ylim(0, 30)
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(all_fitness, label='mean fitness of population', color='green', linestyle='-')
plt.title('Fitness of Kalman Filter for Particle Tracking with EA')
plt.xlabel('Generation')
plt.ylabel('Mean fitness')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(best_individual_fitness, label='fitness of the best individual', color='blue', linestyle='-')
plt.title('Fitness of Kalman Filter for Particle Tracking with EA')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.show()
