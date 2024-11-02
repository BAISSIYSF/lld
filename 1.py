import cv2
import numpy as np

# Callback function for trackbars (does nothing but is needed for trackbars)
def nothing(x):
    pass

# Load the image
image = cv2.imread('path_to_image.jpg')
image = cv2.resize(image, (1280, 720))  # Resize to 720p

# Convert the image to HLS and HSV
hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a window
cv2.namedWindow('Filtered Image')

# Create trackbars for HLS thresholds
cv2.createTrackbar('H_min', 'Filtered Image', 0, 179, nothing)
cv2.createTrackbar('H_max', 'Filtered Image', 179, 179, nothing)
cv2.createTrackbar('L_min', 'Filtered Image', 0, 255, nothing)
cv2.createTrackbar('L_max', 'Filtered Image', 255, 255, nothing)
cv2.createTrackbar('S_min', 'Filtered Image', 0, 255, nothing)
cv2.createTrackbar('S_max', 'Filtered Image', 255, 255, nothing)

# Create trackbars for HSV thresholds
cv2.createTrackbar('V_min', 'Filtered Image', 0, 255, nothing)
cv2.createTrackbar('V_max', 'Filtered Image', 255, 255, nothing)

while True:
    # Get current positions of the trackbars for HLS
    h_min = cv2.getTrackbarPos('H_min', 'Filtered Image')
    h_max = cv2.getTrackbarPos('H_max', 'Filtered Image')
    l_min = cv2.getTrackbarPos('L_min', 'Filtered Image')
    l_max = cv2.getTrackbarPos('L_max', 'Filtered Image')
    s_min = cv2.getTrackbarPos('S_min', 'Filtered Image')
    s_max = cv2.getTrackbarPos('S_max', 'Filtered Image')

    # Get current positions of the trackbars for HSV
    v_min = cv2.getTrackbarPos('V_min', 'Filtered Image')
    v_max = cv2.getTrackbarPos('V_max', 'Filtered Image')

    # Apply thresholds to the HLS image
    hls_mask = cv2.inRange(hls, (h_min, l_min, s_min), (h_max, l_max, s_max))

    # Apply thresholds to the HSV image (only V-channel is relevant here)
    v_channel = hsv[:, :, 2]
    v_mask = cv2.inRange(v_channel, v_min, v_max)

    # Combine HLS and HSV masks
    combined_mask = cv2.bitwise_and(hls_mask, v_mask)

    # Apply the mask to the original image
    filtered_image = cv2.bitwise_and(image, image, mask=combined_mask)

    # Show the filtered image
    cv2.imshow('Filtered Image', filtered_image)

    # Exit on pressing 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cv2.destroyAllWindows()
