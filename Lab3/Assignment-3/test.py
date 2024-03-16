import numpy as np
import matplotlib.pyplot as plt

# Parameters for the first Gaussian distribution
mu1 = 30
sigma1 = 8
weight1 = 0.6  # Weight of the first distribution

# Parameters for the second Gaussian distribution
mu2 = 165
sigma2 = 20
weight2 = 0.4  # Weight of the second distribution

# Generate samples from the two Gaussian distributions
num_samples = 1000
samples1 = np.random.normal(mu1, sigma1, int(num_samples * weight1))
samples2 = np.random.normal(mu2, sigma2, int(num_samples * weight2))

# Combine the samples from both distributions
samples = np.concatenate((samples1, samples2))

# Plot the histogram
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')

# Add a legend
plt.legend(['Double Gaussian Distribution'])

# Show the plot
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Histogram with Double Gaussian Distribution')
plt.show()
