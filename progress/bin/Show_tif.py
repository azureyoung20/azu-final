from PIL import Image
import matplotlib.pyplot as plt

# Open the image file
img = Image.open('E:/CAMELYON16/testmask_001.tif')

# Convert the image to an array for matplotlib to handle
img_arr = np.array(img)

# Create a new matplotlib figure
plt.figure(figsize=(10,10))

# Display the image
plt.imshow(img_arr)

# Show the plot
plt.show()
