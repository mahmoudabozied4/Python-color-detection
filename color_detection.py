import cv2
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------

csv_path = 'colors.csv'

# Reading csv file
index = ['color', 'color_name', 'hex', 'R', 'G', 'B']
df = pd.read_csv(csv_path, names=index, header=None)

# Load the image
img = cv2.imread("/home/zied/img/rgb.jpg")

# Convert image from BGR to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Add additional code to process the image if needed
if img is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully.")

# Define HSV color ranges for multiple colors
color_ranges = {
    "Red": [(np.array([0, 120, 70]), np.array([10, 255, 255])), 
            (np.array([170, 120, 70]), np.array([180, 255, 255]))],
    "Green": [(np.array([35, 100, 100]), np.array([85, 255, 255]))],
    "Blue": [(np.array([100, 150, 0]), np.array([140, 255, 255]))],
    "Yellow": [(np.array([20, 100, 100]), np.array([40, 255, 255]))],
    "Orange": [(np.array([10, 100, 100]), np.array([20, 255, 255]))],
}

# Create a mask and extract colors for each color in the color_ranges dictionary
detected_colors = {}

for color_name, ranges in color_ranges.items():
    color_mask = None
    
    # Combine masks for colors that span multiple HSV ranges (e.g., Red)
    for lower, upper in ranges:
        mask = cv2.inRange(hsv_img, lower, upper)
        color_mask = mask if color_mask is None else color_mask | mask
    
    # Bitwise AND to extract the color areas from the original image
    color_detected = cv2.bitwise_and(img, img, mask=color_mask)
    
    # Store the result
    detected_colors[color_name] = color_detected

# Display the original image and the color detection results
cv2.imshow('Original Image', img)

# Display images for each detected color
for color_name, detected in detected_colors.items():
    cv2.imshow(f'{color_name} Detection', detected)

# Wait for user input to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
