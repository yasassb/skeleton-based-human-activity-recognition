import json
import numpy as np
import os
import cv2

# Function to get the dimensions (width and height) of a video file
def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)  # Open the video file
    if not cap.isOpened():  # Check if the video file was successfully opened
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Retrieve the width and height of the video frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()  # Release the video capture object
    return width, height

# Function to normalize keypoints based on the frame dimensions
def normalize_keypoints(keypoints, frame_width, frame_height):
    normalized = []  # List to store normalized keypoints
    for kp in keypoints:
        # Normalize x and y coordinates by dividing by frame width and height
        x = kp[0] / frame_width
        y = kp[1] / frame_height
        normalized.append([x, y])  # Append normalized keypoint
    return normalized

# Function to process a JSON file and normalize its keypoints
def process_json(input_path, output_path, frame_width, frame_height):
    with open(input_path, 'r') as f:  # Open the input JSON file
        data = json.load(f)  # Load the JSON data
    
    # Process each entry in the JSON file
    for entry in data:
        if isinstance(entry, list):  # If the entry is a list of sub-entries
            for sub_entry in entry:
                # Normalize keypoints for each sub-entry
                sub_entry["keypoints"] = normalize_keypoints(
                    sub_entry["keypoints"], 
                    frame_width, 
                    frame_height
                )
        else:  # If the entry is a single object
            # Normalize keypoints for the entry
            entry["keypoints"] = normalize_keypoints(
                entry["keypoints"], 
                frame_width, 
                frame_height
            )
    
    # Save the normalized data to a new JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

# Function to process all JSON files in a given folder and its subfolders
def process_all_json_files(main_folder):
    for root, dirs, files in os.walk(main_folder):  # Walk through the folder structure
        for file in files:  # Iterate over all files
            if file.endswith('.json'):  # Process only JSON files
                json_path = os.path.join(root, file)  # Get the full path of the JSON file
                
                # Find the corresponding video file (same name but with .mp4 extension)
                video_name = file.replace('.json', '.mp4')
                video_path = os.path.join(root, video_name)
                
                if not os.path.exists(video_path):  # Check if the video file exists
                    print(f"Warning: No corresponding video found for {json_path}")
                    continue  # Skip processing if the video file is missing
                
                try:
                    # Get the dimensions of the video
                    width, height = get_video_dimensions(video_path)
                    
                    # Create the output path for the normalized JSON file
                    output_filename = file.replace('.json', '_normalized.json')
                    output_path = os.path.join(root, output_filename)
                    
                    # Process the JSON file and normalize its keypoints
                    process_json(json_path, output_path, width, height)
                    print(f"Processed: {json_path} -> {output_path}")
                    
                except Exception as e:  # Handle any errors during processing
                    print(f"Error processing {json_path}: {str(e)}")

# Example usage of the script
main_folder = "keypoints"  # Main folder containing subfolders with JSON and video files
process_all_json_files(main_folder)  # Process all JSON files in the folder
print("Normalization complete for all files.")  # Print completion message