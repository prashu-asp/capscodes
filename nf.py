import cv2

# Load the image
image = cv2.imread('F:\Focus and Non-Focus\sample images//nf1.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours to find squares
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:  # Check for quadrilateral
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:  # Check for square
            # Draw the contour on the original image
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 5)
            # Optionally, break if only the first square is needed
            break

# Display the result
cv2.imshow('Detected Square', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
