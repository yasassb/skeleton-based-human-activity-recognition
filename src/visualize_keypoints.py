import json
import cv2
import numpy as np
import os

# Load JSON
# with open('keypoints/Meet and Split/Meet and Split (1).json', 'r') as f:
with open('keypoints/Walking/Walking (1)_normalized.json', 'r') as f:
    data = json.load(f)

# Output settings
frame_size = (2000, 2000)  # Width x Height
fps = 30 

# Skeleton connections (using COCO format)
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

# Iterate through each frame in the dataset
for idx, frame_data in enumerate(data):
    # Create a blank image for visualization
    frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)  # Black background

    # Iterate through all people in the current frame
    for person_data in frame_data:
        keypoints = person_data['keypoints']  # Extract keypoints for the current person

        # Draw keypoints for the current person
        for x, y in keypoints:
            x, y = x * 1000, y * 1000   # Comment to see the keypoints in the original scale
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green keypoints

        # Draw skeleton for the current person
        for i, j in skeleton:
            if i < len(keypoints) and j < len(keypoints):
                x1, y1 = keypoints[i]
                x2, y2 = keypoints[j]
                x1, y1 = x1 * 1000, y1 * 1000   # Comment to see the keypoints in the original scale
                x2, y2 = x2 * 1000, y2 * 1000   # Comment to see the keypoints in the original scale
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue lines

    # Display the frame using OpenCV
    cv2.imshow('Real-time Keypoint Visualization', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
print("Real-time visualization completed.")