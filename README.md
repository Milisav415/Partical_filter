🌀 Particle Filter Algorithm
Description
The Particle Filter (also known as Sequential Monte Carlo method) is a powerful and widely used algorithm for estimating the state of a system based on noisy observations over time. It is particularly useful for dynamic systems that are non-linear and subject to non-Gaussian noise, where traditional filtering methods like the Kalman filter may not be applicable.

In this implementation, the Particle Filter algorithm is used to estimate the state of a system by approximating the posterior distribution using a set of weighted particles. These particles represent potential states of the system, and the algorithm updates them as new observations are received.

How It Works
Initialization:

A set of particles is initialized, each representing a possible state of the system.
Each particle is assigned an initial weight, typically equal for all particles at the beginning.
Prediction Step:

The state of each particle is updated based on the system's dynamic model. This step predicts the next state of the particles by applying a process model (e.g., motion or transition model) to each particle.
Update Step (Correction):

Once new measurements or observations are available, the particles are "resampled" based on their likelihood to match the observations.
The weights of the particles are updated based on how well the predicted state of each particle aligns with the observed data.
Resampling:

Particles with higher weights (indicating better fit to the observations) are more likely to be retained in the next iteration.
This step helps in focusing on the regions of the state space that have a higher probability of matching the observations.
Estimate State:

The final state estimate is typically the weighted average of all the particles.
This provides an estimate that accounts for all possible states, weighted by their likelihood.
Key Features
Non-linear and Non-Gaussian: The Particle Filter is ideal for systems that do not fit the assumptions of traditional linear models (like the Kalman filter).
Robust to Noise: It can handle systems with high levels of uncertainty or noise in both the process model and measurements.
Multiple Hypothesis Tracking: It is capable of tracking multiple possible states of the system simultaneously, which is useful for complex environments with ambiguity.
Applications
Localization and Mapping: Widely used in robotics for estimating a robot's position in an environment.
Tracking: Used for object tracking in video processing or sensor networks.
Navigation Systems: Applied in GPS-based navigation and self-driving cars.
Signal Processing: Used in signal processing for filtering noisy data.
