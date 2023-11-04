import cv2
import numpy as np

# Load the image
image = cv2.imread("folder/WhatsApp Image 2023-11-04 at 9.48.23 PM.jpeg")

# Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow(gray)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Threshold the image to create a binary mask
_, thresholded = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

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
