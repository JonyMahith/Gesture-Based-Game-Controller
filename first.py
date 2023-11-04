import cv2
import numpy as np

# Load the image from the file
img = cv2.imread('folder/finger scan.jpg')

# Check if the image was loaded successfully
if img is not None:
    # Define the desired width and height for the resized image
    new_width = 800  # Change this value to your desired width
    new_height = 600  # Change this value to your desired height

    # Resize the image to the new dimensions
    image = cv2.resize(cake, (new_width, new_height))

    # Display the resized image in a window
    cv2.imshow('image', image)

    # Wait for a key press indefinitely
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()
else:
    print("Error: Could not load the image.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image to create a binary mask
_, thresholded = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

# Convert the thresholded image to CV_8UC1 format
thresholded = thresholded.astype(np.uint8)

# Find contours in the binary mask
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a variable to count fingers
finger_count = 0

# Define a region of interest (ROI) where you expect fingers
roi = image[100:300, 100:300]

# Loop through the contours and check if they are within the ROI
for contour in contours:
    if cv2.pointPolygonTest(contour, (200, 200), False) > 0:
        finger_count += 1

# Display the image with the finger count
cv2.putText(image, f'Fingers: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow('Finger Count', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f'Number of fingers: {finger_count}')