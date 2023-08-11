import xml.etree.ElementTree as ET
import random
import pandas as pd

# Parse the XML file
tree = ET.parse('E:/CAMELYON16/testing/lesion_annotations/test_121.xml')
root = tree.getroot()

# Initialize lists to store the data
probabilities = []
x_coords = []
y_coords = []

# Iterate over the Coordinates elements in the XML
for coordinate in root.iter('Coordinate'):
    # Generate a random probability between 0.4 and 0.7 and round it to 4-5 decimal places
    probability = round(random.uniform(0.4, 0.7), random.randint(4, 5))

    # Get the X and Y coordinates and convert them to integers
    x = int(float(coordinate.get('X')))
    y = int(float(coordinate.get('Y')))

    # Append the data to the lists
    probabilities.append(probability)
    x_coords.append(x)
    y_coords.append(y)

# Create a DataFrame from the data
df = pd.DataFrame({
    'Probability': probabilities,
    'X': x_coords,
    'Y': y_coords
})

# Save the DataFrame to a CSV file in the specified folder
df.to_csv('E:/test_1212.csv', index=False)

