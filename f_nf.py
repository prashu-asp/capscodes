
import cv2
import numpy as np

# Load the image
image = cv2.imread('F:\Focus and Non-Focus\sample images//nf1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection
edges = cv2.Canny(blurred, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:  # Check for quadrilateral
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:  # Likely a square

            # Calculate mean intensity within the square
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [approx], -1, 255, -1)
            mean_inside = cv2.mean(gray, mask=mask)[0]
            
            # Calculate mean intensity along the border
            border_mask = np.zeros_like(gray)
            cv2.drawContours(border_mask, [approx], -1, 255, 3)  # Adjust thickness as needed
            mean_border = cv2.mean(gray, mask=border_mask)[0]

            # Check if there's a significant difference between the border and the inside
            if abs(mean_border - mean_inside) > 30:  # Threshold may need adjustment
                print("Highlighted Square Detected")
                cv2.drawContours(image, [approx], 0, (0, 255, 0), 5)  # Green border for highlighted square
            else:
                print("Normal Square Detected")
                cv2.drawContours(image, [approx], 0, (255, 0, 0), 5)  # Blue border for normal square

# Display the result
cv2.imshow('Squares Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
