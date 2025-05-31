import os
import json
import matplotlib.pyplot as plt

# Path to the root directory containing JSON files
keypoints_root = "keypoints"

def count_frames_in_json(json_path):
    """Count the number of frames in a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)  # Load the JSON data
    return len(data)  # Return the number of frames (entries in the JSON)

def analyze_frame_counts(root_dir):
    """Analyze frame counts across all JSON files in the directory."""
    frame_counts = []  # List to store frame counts for each file

    # Walk through the directory and its subdirectories
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):  # Process only JSON files
                json_path = os.path.join(subdir, file)  # Get the full path of the JSON file
                frame_count = count_frames_in_json(json_path)  # Count frames in the JSON file
                frame_counts.append((json_path, frame_count))  # Append the file path and frame count

    if not frame_counts:  # If no JSON files were found
        print("No JSON files found.")
        return

    # Find the file with the maximum and minimum frame counts
    max_file, max_count = max(frame_counts, key=lambda x: x[1])  # File with the maximum frame count
    min_file, min_count = min(frame_counts, key=lambda x: x[1])  # File with the minimum frame count

    # Print the results
    print(f"Maximum frames: {max_count} in file {max_file}")
    print(f"Minimum frames: {min_count} in file {min_file}")

    # Plot the frame counts
    plot_frame_counts(frame_counts)

def plot_frame_counts(frame_counts):
    """Plot frame counts against file indices."""
    indices = range(len(frame_counts))  # Use indices as x-axis values
    frame_values = [count for _, count in frame_counts]  # Extract frame counts

    # Create a line graph
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(indices, frame_values, marker='o', color='skyblue', linestyle='-')  # Line graph with markers
    plt.xlabel('File Index')  # Label for the x-axis
    plt.ylabel('Frame Count')  # Label for the y-axis
    plt.title('Frame Counts per File')  # Title of the graph
    plt.grid(True)  # Add grid for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()  # Display the plot

# Run the analysis
analyze_frame_counts(keypoints_root)