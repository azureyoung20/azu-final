import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

# Load the numpy array from a file
t051 = np.load('F:/Dissertation/论文撰写/实验结果材料/test_051baseline72081.npy')

# Transpose the numpy array
t051_transposed = t051.T

# Visualize the transposed array
plt.imshow(t051_transposed)
plt.savefig('t051_transposed.jpg')
plt.show()

# Save the transposed array as an image file
scipy.misc.imsave("t051_transposed.png", t051_transposed)

