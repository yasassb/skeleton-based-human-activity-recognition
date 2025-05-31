# Import necessary libraries
import os
import subprocess
from pathlib import Path

# Configuration
input_root = "/Users/yasas/Documents/coursework-comp5013-panicatthekernel/Human_Activity_Recognition_Video_Dataset"  # Input directory containing video files
output_root = "/Users/yasas/Documents/coursework-comp5013-panicatthekernel/keypoints"  # Output directory for storing results
inferencer_script = "src/inferencer.py"  # Path to the inferencer script
pose2d_method = "rtmo"  # Pose estimation method to be used

# Create the output root directory if it doesn't exist
Path(output_root).mkdir(parents=True, exist_ok=True)

# Walk through the input directory recursively
for root, dirs, files in os.walk(input_root):
    for file in files:
        # Check if the file is a video file based on its extension
        if file.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):  # Supported video formats
            input_video = os.path.join(root, file)  # Full path to the input video
            
            # Create a relative path for the output directory based on the input directory structure
            relative_path = os.path.relpath(root, input_root)
            output_dir = os.path.join(output_root, relative_path)
            
            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Build the command to run the inferencer script
            cmd = [
                "python",  # Python interpreter
                inferencer_script,  # Path to the inferencer script
                input_video,  # Input video file
                "--pose2d", pose2d_method,  # Specify the pose estimation method
                "--vis-out-dir", output_dir,  # Directory to save visualized output
                #"--show",  # Uncomment to display the output during processing
                "--draw-bbox",  # Enable drawing bounding boxes
                "--bbox-thr", "0.6",  # Bounding box threshold
                "--device", "mps"  # Specify the device (e.g., Metal Performance Shaders on macOS)
            ]
            
            # Print progress information
            print(f"\nProcessing: {input_video}")
            print(f"Saving to: {output_dir}")
            
            # Execute the command using subprocess
            subprocess.run(cmd)

# Print a message indicating that processing is complete
print("\nProcessing complete! Check your output directory:", output_root)