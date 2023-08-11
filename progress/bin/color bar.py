import matplotlib.pyplot as plt
import numpy as np

# Create a gradient image that goes from 0 to 1
gradient = np.linspace(0, 1, 256)  # 256 is the number of color levels in the color map
gradient = np.vstack((gradient, gradient))  # Make the gradient image 2D

fig, ax = plt.subplots(figsize=(6, 1))  # Adjust the size to fit your needs
fig.subplots_adjust(bottom=0.5)

# Display the gradient image with the 'jet' color map
cax = ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap('jet'))

# Remove the plot frame and ticks
ax.axis('off')

# Add a color bar
cbar = fig.colorbar(cax, ticks=[0, 1], orientation='horizontal')
cbar.ax.set_xticklabels(['0 (Blue)', '1 (Yellow)'])  # Set color bar labels

plt.show()
