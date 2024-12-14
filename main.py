import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse

matplotlib.use('Qt5Agg')

# Load the data from the CSV file
file_path = 'C:/Users/jm190/Desktop/VI/dz_2/observations.csv'
observations_df = pd.read_csv(file_path)


# Convert observations from polar to Cartesian
def polar_to_cartesian(r, theta):
    """
    Converts polar coordinates to cartesian coordinates
    :param r: polar radius
    :param theta: polar angle
    :return: cartesian coordinates
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


# convert to cartesian
observations_cartesian = np.array([polar_to_cartesian(r, theta) for r, theta in observations_df.to_numpy()])

# Initialize constants and parameters
num_particles = 2000
velocity = 0.5
direction_change = [0 for _ in range(num_particles)]  # Helper list for the measurement model
direction_change_probs = [0.2, 0.4, 0.6, 0.8, 1.0]
direction_change_range = np.pi / 6

# Initialize particles with random positions and directions
particles = np.zeros((num_particles, 3))  # Each particle has (x, y, direction)
particles[:, :2] = np.random.uniform(-2, 2, size=(num_particles, 2))
particles[:, 2] = np.random.uniform(-np.pi, np.pi, num_particles)

# Initialize weights
weights = np.array([1/num_particles for _ in range(num_particles)])


# Predict step
def predict(particles, velocity):
    """
    Predict the position of the robot and change the velocity angle of all the particles
    :param particles: particles
    :param velocity: intensity of the velocity vector
    :return: predicted position and the particles in one form
    """
    for i, particle in enumerate(particles):
        particle[0] += velocity * np.cos(particle[2])  # Update x
        particle[1] += velocity * np.sin(particle[2])  # Update y
        change_prob = np.random.rand()  # 0->1
        if change_prob < direction_change_probs[direction_change[i]]:
            particle[2] += np.random.uniform(-direction_change_range, direction_change_range)
            direction_change[i] = 0  # reset counter
        else:
            direction_change[i] += 1  # increment counter
    return particles


# Refined update step
def update_weights_refined(particles, observations_r, observations_theta, std_dev_distance, std_dev_angle):
    """
    Updates the weights of the particles.
    :param particles: The particles.
    :param observations_r: Observations po.
    :param observations_theta: Observations theta.
    :param std_dev_distance: Standard deviation of distance.
    :param std_dev_angle: standard deviation of angle.
    :return: Updated wights.
    """
    global weights
    for i, particle in enumerate(particles):
        predicted_r = np.linalg.norm(particle[:2])
        predicted_theta = np.arctan2(particle[1], particle[0])
        distance_error = observations_r - predicted_r
        angle_error = observations_theta - predicted_theta
        distance_likelihood = np.exp(-(distance_error ** 2) /
                                     (2 * ((2 - np.abs(np.cos(particle[2]))) * std_dev_distance) ** 2))
        angle_likelihood = np.exp(-(np.abs(angle_error) / std_dev_angle))
        weights[i] *= distance_likelihood * angle_likelihood
    weights /= np.sum(weights)
    return weights


def reset_weights():
    """
    Resets the weights of the particles.
    :return: void
    """
    global weights
    weights = np.array([1 / num_particles for _ in range(num_particles)])


def resample(particles):
    """
    Resample step takes the biggest particle and split it
    :param particles: The particles to resample
    :param weights: The weights of the particles
    :return: The resampled particles list
    """
    global weights
    indices = np.random.choice(range(len(particles)), size=len(particles), p=weights)
    reset_weights()
    return particles[indices]


# Function to draw an ellipse representing the covariance
def plot_cov_ellipse(cov, pos, nstd=2, **kwargs):
    """
    Plot covariance ellipse, or interval of trust
    :param cov: covariance matrix
    :param pos: position of ellipse
    :param nstd: standard deviation of ellipse
    :param kwargs: keyword arguments
    :return: void
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)


# Estimate step
def estimate(particles):
    """
    Estimate the co-ordinates of the true position of the robot
    :param particles:
    :param weights:
    :return: mean of positions X, Y and angle of speed vector
    """
    global weights
    weights_norm = weights / np.max(weights)  # normalize the weights

    mean_x = np.sum(particles[:, 0] * weights)
    mean_y = np.sum(particles[:, 1] * weights)
    mean_angle = np.sum(particles[:, 2] * weights)

    return mean_x, mean_y, mean_angle


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Parameters for the measurement model
    std_dev_distance = 0.3  # Placeholder value
    std_dev_angle = np.pi / 36

    # Initialize particles
    particles = np.zeros((num_particles, 3))
    particles[:, :2] = np.random.uniform(-2, 2, size=(num_particles, 2))
    particles[:, 2] = np.random.uniform(-np.pi, np.pi, num_particles)

    # Initialize arrays to store heavy particles and their weights
    top_n = 100
    heavy_particles = []
    heavy_weights = []
    estimated_positions = []

    # set an empty list for weight tracking
    weights_over_time = []

    # Refined particle filter with tracking of top 5 particles
    all_estimates = []
    measured_r, measured_theta = observations_df.iloc[:, 0], observations_df.iloc[:, 1]
    cnt = 0  # counter tracking the number of iterations
    for observation_r, observation_theta in zip(measured_r, measured_theta):
        particles = predict(particles, velocity)
        weights = update_weights_refined(particles, observation_r, observation_theta, std_dev_distance, std_dev_angle)

        # Store weights for plotting
        weights_over_time.append(weights.copy())

        # Normalize weights for transparency on plot
        weights_normalized = weights / np.max(weights)

        heavy_idx = np.argsort(weights)[-top_n:]
        heavy_particles = particles[heavy_idx]
        heavy_weights = weights[heavy_idx]

        # estimate position
        estimate_pos = estimate(particles)

        # Calculate standard deviation region
        estimated_positions_np = np.array(estimated_positions)
        cov = np.cov(particles[:, 0], particles[:, 1])

        # Resample the particles, allways do it last!!!
        particles = resample(particles)  # delete this line to disable resampling !!!

        # Collect the estimated positions
        estimated_positions.append(estimate_pos)
        all_estimates.append(estimate_pos)

        measured_cartesian = polar_to_cartesian(observation_r, observation_theta)
        heavy_particles_cartesian = [(p[0], p[1]) for p in heavy_particles]

        plt.figure(figsize=(8, 6))
        plt.scatter(*measured_cartesian, color='blue', label='Measured Position', marker='*', s=80)
        for i, p in enumerate(heavy_particles_cartesian):
            plt.scatter(*p, color='green', alpha=weights_normalized[i])
        plt.scatter(estimate_pos[0], estimate_pos[1], color='red', label='Estimated Position', s=100)

        # Calculate standard deviation region
        estimated_positions_np = np.array(estimated_positions)
        std_x = np.std(estimated_positions_np[:, 0])
        std_y = np.std(estimated_positions_np[:, 1])

        # Draw the 2σ confidence ellipse
        ellipse = plot_cov_ellipse(cov, estimate_pos, nstd=2, color='orange', alpha=0.3, label='2σ region')
        plt.gca().add_patch(ellipse)

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.title(f'Measurement, Top 100 Particles, and Estimated Position for measurement: {cnt}')
        plt.grid(True)
        plt.show()
        cnt += 1  # increment counter

    # Convert final estimates to polar coordinates for plotting the 2*std region
    final_estimates_df = pd.DataFrame(all_estimates, columns=["x", "y", "angle"])
    final_estimated_r = np.sqrt(final_estimates_df["x"] ** 2 + final_estimates_df["y"] ** 2)
    final_estimated_theta = np.arctan2(final_estimates_df["y"], final_estimates_df["x"])

    # Calculate the mean and 2*std of the final estimates
    mean_r = np.mean(final_estimated_r)
    std_r = np.std(final_estimated_r)

    # Convert observed and estimated polar coordinates to Cartesian
    observed_x, observed_y = zip(*observations_cartesian)
    estimated_x, estimated_y = final_estimates_df["x"], final_estimates_df["y"]

    # Plot the observations and estimates in Cartesian coordinates
    plt.figure(figsize=(10, 6))
    plt.scatter(observed_x, observed_y, color='blue', label='Measured Positions', alpha=0.5)
    plt.plot(observed_x, observed_y, color='blue', label='Measured path', linewidth=2)
    plt.scatter(estimated_x, estimated_y, color='red', label='Estimated Positions', alpha=0.5)
    plt.plot(estimated_x, estimated_y, color='red', label='Estimated path', linewidth=2)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.title('Measured vs Estimated Positions (Cartesian Coordinates)')
    plt.grid(True)
    plt.show()

    # Plot weights over iterations
    weights_over_time = np.array(weights_over_time)
    plt.figure(figsize=(10, 6))
    for i in range(num_particles):
        plt.plot(range(cnt), weights_over_time[:, i], alpha=0.5)
    plt.xlabel('Time Steps')
    plt.ylabel('Weight')
    plt.title('Weights Over Iterations')
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
