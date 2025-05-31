'''
GitHub: https://github.com/open-mmlab/mmpose

Prerequisites:
- install mmpose and related packages
- download the config and the checkpoint of the model
    mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest .
'''

# Copyright (c) OpenMMLab. All rights reserved.

# Import necessary libraries
from argparse import ArgumentParser
from typing import Dict
from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases  # MMPose API for pose estimation
import json
import numpy as np
from tqdm import tqdm  # For progress bar
import os
import cv2
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.preprocessing import LabelEncoder
import math
from collections import Counter
# Import model architecture from training script
from cnn_transformer_train import Config, SpatialAttention, PositionalEncoding, CNNTransformerModel

# Default filtering arguments for pose estimation
# These parameters control detection quality and filtering
filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)

# Model-specific filtering arguments to optimize detection
# Different pose models have different optimal filtering parameters
POSE2D_SPECIFIC_ARGS = dict(
    yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
    rtmo=dict(bbox_thr=0.1, nms_thr=0.65, pose_based_nms=True),
)

# Function to normalize keypoints based on image dimensions
# Converts absolute pixel coordinates to relative [0,1] range
def normalize_keypoints(keypoints, width, height):
    keypoints_copy = keypoints.copy()
    # X coordinates (even indices)
    keypoints_copy[0::2] = keypoints_copy[0::2] / width
    # Y coordinates (odd indices)
    keypoints_copy[1::2] = keypoints_copy[1::2] / height
    return keypoints_copy

# Function to parse command-line arguments
# Sets up all configurable parameters for the inference process
def parse_args():
    parser = ArgumentParser()

    # Activity recognition specific parameters
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.3,
        help='Threshold for keypoint confidence')
        
    parser.add_argument(
        '--smooth-window',
        type=int,
        default=5,
        help='Window size for temporal smoothing of predictions')

    parser.add_argument(
        '--model-path', 
        type=str, 
        required=True,
        help='Path to trained model weights')
    
    parser.add_argument(
        '--label-encoder', 
        type=str, 
        required=True,
        help='Path to label encoder classes')

    # Input path for image/video or folder
    parser.add_argument(
        'inputs',
        type=str,
        nargs='?',
        help='Input image/video path or folder path.')

    # init args for MMPose
    # Arguments for 2D pose estimation
    parser.add_argument(
        '--pose2d',
        type=str,
        default=None,
        help='Pretrained 2D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose2d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose2d" is a model name of metafile, '
        'the weights will be loaded from metafile.')

    # Arguments for 3D pose estimation (if needed)
    parser.add_argument(
        '--pose3d',
        type=str,
        default=None,
        help='Pretrained 3D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose3d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose3d" is a model name of metafile, '
        'the weights will be loaded from metafile.')

    # Arguments for person detection model
    parser.add_argument(
        '--det-model',
        type=str,
        default=None,
        help='Config path or alias of detection model.')
    parser.add_argument(
        '--det-weights',
        type=str,
        default=None,
        help='Path to the checkpoints of detection model.')
    parser.add_argument(
        '--det-cat-ids',
        type=int,
        nargs='+',
        default=0,
        help='Category id for detection model.')

    # Other general arguments for MMPose
    parser.add_argument(
        '--scope',
        type=str,
        default='mmpose',
        help='Scope where modules are defined.')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    parser.add_argument(
        '--show-progress',
        action='store_true',
        help='Display the progress bar during inference.')

    # The default arguments for prediction filtering differ for top-down
    # and bottom-up models. We assign the default arguments according to the
    # selected pose2d model
    args, _ = parser.parse_known_args()
    for model in POSE2D_SPECIFIC_ARGS:
        if args.pose2d is not None and model in args.pose2d:
            filter_args.update(POSE2D_SPECIFIC_ARGS[model])
            break

    # Visualization and output control arguments
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image/video in a popup window.')
    parser.add_argument(
        '--draw-bbox',
        action='store_true',
        help='Whether to draw the bounding boxes.')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Whether to draw the predicted heatmaps.')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=filter_args['bbox_thr'],
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=filter_args['nms_thr'],
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--pose-based-nms',
        type=lambda arg: arg.lower() in ('true', 'yes', 't', 'y', '1'),
        default=filter_args['pose_based_nms'],
        help='Whether to use pose-based NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--use-oks-tracking',
        action='store_true',
        help='Whether to use OKS as similarity in tracking')
    parser.add_argument(
        '--disable-norm-pose-2d',
        action='store_true',
        help='Whether to scale the bbox (along with the 2D pose) to the '
        'average bbox scale of the dataset, and move the bbox (along with the '
        '2D pose) to the average bbox center of the dataset. This is useful '
        'when bbox is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--disable-rebase-keypoint',
        action='store_true',
        default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
        'lowest keypoint has a height of 0 (landing on the ground). Rebase '
        'is useful for visualization when the model do not predict the '
        'global position of the 3D pose.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization.')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization.')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--black-background',
        action='store_true',
        help='Plot predictions on a black image')
    parser.add_argument(
        '--vis-out-dir',
        type=str,
        default='',
        help='Directory for saving visualized results.')
    parser.add_argument(
        '--pred-out-dir',
        type=str,
        default='',
        help='Directory for saving inference results.')
    parser.add_argument(
        '--show-alias',
        action='store_true',
        help='Display all the available model aliases.')

    # Parse arguments and separate initialization and call arguments
    call_args = vars(parser.parse_args())

    # Extract custom arguments for activity recognition
    custom_args = {
        'confidence_threshold': call_args.pop('confidence_threshold'),
        'smooth_window': call_args.pop('smooth_window'),
        'model_path': call_args.pop('model_path'),
        'label_encoder': call_args.pop('label_encoder')
    }

    # Separate initialization arguments for MMPose
    init_kws = [
        'pose2d', 'pose2d_weights', 'scope', 'device', 'det_model',
        'det_weights', 'det_cat_ids', 'pose3d', 'pose3d_weights',
        'show_progress'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    # Combine custom args with MMPose init args
    init_args.update(custom_args)

    display_alias = call_args.pop('show_alias')

    return init_args, call_args, display_alias

def main():
    # Parse all arguments for pose estimation and activity recognition
    init_args, call_args, display_alias = parse_args()
    
    # Extract activity recognition specific parameters
    confidence_threshold = init_args.pop('confidence_threshold')
    smooth_window = init_args.pop('smooth_window')
    
    # Verify that required files exist
    if not os.path.exists(init_args['model_path']):
        raise FileNotFoundError(f"Model file not found: {init_args['model_path']}")
    if not os.path.exists(init_args['label_encoder']):
        raise FileNotFoundError(f"Label encoder file not found: {init_args['label_encoder']}")
    
    # Load label encoder with class names for activity recognition
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(init_args['label_encoder'], allow_pickle=True)
    
    # Initialize model configuration with the same parameters used during training
    config = Config()
    config.num_classes = len(label_encoder.classes_)
    device = config.device  # Use device from config (GPU if available)
    
    # Load the pretrained CNN-Transformer model
    model = CNNTransformerModel(config).to(device)
    model.load_state_dict(torch.load(init_args['model_path'], map_location=device))
    model.eval()  # Set to evaluation mode
    
    print("Model loaded successfully!")
    print(f"Activity classes: {list(label_encoder.classes_)}")
    
    # Get input video/image path
    video_path = call_args.get('inputs', '')
    
    # Get video or image metadata (dimensions and total frames)
    width, height, total_frames = 0, 0, 0
    if video_path.endswith(('.mp4', '.avi')):
        # Video case - get dimensions and frame count
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    elif os.path.isfile(video_path):  # Single image case
        img = cv2.imread(video_path)
        if img is not None:
            height, width, _ = img.shape
            total_frames = 1
    else:
        raise ValueError("Invalid input path")
    
    # Create a copy of init_args without activity recognition arguments
    # to pass only pose estimation related parameters to MMPose
    pose_init_args = init_args.copy()
    pose_init_args.pop('model_path', None)
    pose_init_args.pop('label_encoder', None)
    
    # Initialize MMPose inferencer with only pose estimation arguments
    # This creates the pose detection model used to extract keypoints from each frame
    inferencer = MMPoseInferencer(**pose_init_args)
    
    # Buffer to store keypoint sequences for activity recognition
    sequence_buffer = []
    sequence_length = config.chunk_size  # Match training chunk size (64 frames)
    overlap = config.overlap  # Match training overlap (16 frames) for sliding window approach
    prev_kps_flat = None  # Track previous frame's keypoints to calculate velocity features
    results = []  # Store all activity recognition results
    recent_predictions = []  # Buffer for temporal smoothing of predictions
    prediction_history_size = smooth_window  # Size of window for majority voting/smoothing
    
    # Set up output directory and file for saving prediction results
    if call_args.get('pred_out_dir', ''):
        os.makedirs(call_args.get('pred_out_dir', ''), exist_ok=True)
        output_path = os.path.join(call_args.get('pred_out_dir', ''), 'activity_results.json')
    else:
        output_path = 'cnn_transformer/activity_results.json'  # Default output path if none specified
    
    # Process frames with a progress bar to show completion status
    print(f"Processing video with chunk size {sequence_length} and overlap {overlap}")
    with tqdm(total=total_frames, desc="Processing") as pbar:
        # Iterate through frames processed by the pose estimator
        for result in inferencer(**call_args):
            # Check if pose predictions are available for current frame
            if 'predictions' in result and result['predictions']:
                frame_keypoints = []  # Will store processed keypoints for the current frame
                
                # Extract and process keypoints (focus on first person only)
                if result['predictions'][0]:  # Check if any detections in this frame
                    # Get keypoint confidences from MMPose output
                    keypoint_confidences = np.array(result['predictions'][0][0]['keypoint_scores'])
                    
                    # Only process keypoints if average confidence is above threshold
                    # This helps filter out low-quality pose detections
                    if np.mean(keypoint_confidences) >= confidence_threshold:
                        # Extract keypoint coordinates (17 joints, x and y)
                        current_kps = np.array(result['predictions'][0][0]['keypoints'])[:,:2]  # (17, 2)
                        current_kps_flat = current_kps.flatten()  # Flatten to 1D array (34,)
                        
                        # Normalize keypoints to [0,1] range by dividing by image dimensions
                        current_kps_flat = normalize_keypoints(current_kps_flat, width, height)
                        
                        # Calculate velocity (motion between frames) as a feature
                        if prev_kps_flat is None:
                            # First frame - no velocity data
                            velocity = np.zeros_like(current_kps_flat)
                        else:
                            # Calculate the difference in position from previous frame
                            velocity = current_kps_flat - prev_kps_flat
                        
                        # Safety check - replace any NaN or inf values with zeros
                        if not np.isfinite(velocity).all():
                            velocity = np.zeros_like(current_kps_flat)
                        
                        # Concatenate position and velocity to create feature vector
                        # Total 68 features: 34 position (17 joints × 2 coordinates) + 34 velocity
                        kp_with_velocity = np.concatenate([current_kps_flat, velocity])
                        frame_keypoints.append(kp_with_velocity)
                        
                        # Store current keypoints for next frame's velocity calculation
                        prev_kps_flat = current_kps_flat.copy()
                
                # Decide what to add to the sequence buffer
                if frame_keypoints:
                    # If keypoints detected, add them
                    sequence_buffer.append(frame_keypoints[0])
                else:
                    # If no detection in current frame, use last known keypoints with zero velocity
                    # This provides continuity during brief detection losses
                    if prev_kps_flat is not None:
                        zero_velocity = np.zeros_like(prev_kps_flat)
                        kp_with_zero_velocity = np.concatenate([prev_kps_flat, zero_velocity])
                        sequence_buffer.append(kp_with_zero_velocity)
                    else:
                        # If no previous keypoints either, add zeros (empty frame)
                        sequence_buffer.append(np.zeros(config.input_dim))
                
                # Process a complete sequence when buffer reaches required length
                if len(sequence_buffer) >= sequence_length:
                    # Extract exactly sequence_length frames for processing
                    seq = np.array(sequence_buffer[:sequence_length])
                    
                    # Sanity check - ensure no NaN or inf values that could disrupt model
                    if np.isfinite(seq).all():
                        # Run activity recognition model inference
                        with torch.no_grad():
                            # Prepare input tensor
                            inputs = torch.FloatTensor(seq).unsqueeze(0).to(device)
                            # Forward pass through CNN-Transformer model
                            outputs = model(inputs)
                            # Get predicted class index
                            # _, pred = torch.max(outputs, 1)
                            _, pred = torch.max(outputs[0], 1)
                            pred_class = pred.item()
                            
                            # Apply temporal smoothing to stabilize predictions
                            recent_predictions.append(pred_class)
                            if len(recent_predictions) > prediction_history_size:
                                recent_predictions.pop(0)  # Remove oldest prediction
                            
                            # Get most common prediction in the temporal window (majority voting)
                            if recent_predictions:
                                smoothed_pred = Counter(recent_predictions).most_common(1)[0][0]
                                # Convert numerical class to human-readable label
                                activity = label_encoder.inverse_transform([smoothed_pred])[0]
                                # Calculate prediction confidence 
                                # confidence = torch.softmax(outputs, dim=1)[0][pred].item()
                                confidence = torch.softmax(outputs[0], dim=1)[0][pred].item()
                                
                                # Store results with frame range information
                                start_frame = max(0, pbar.n - sequence_length + 1)
                                results.append({
                                    'frames': f"{start_frame}-{pbar.n}",
                                    'activity': activity,
                                    'confidence': float(confidence),
                                    'raw_prediction': int(pred_class),
                                    'smoothed_prediction': int(smoothed_pred)
                                })
                                
                                # Print current prediction
                                print(f"\nFrames {start_frame}-{pbar.n}: {activity} (conf: {confidence:.2f})")
                    else:
                        print("\nSkipping prediction due to invalid values in sequence")
                    
                    # Reset buffer with overlap for sliding window approach
                    # Keep last 'overlap' frames to maintain continuity between chunks
                    sequence_buffer = sequence_buffer[sequence_length - overlap:]
                
                # Update progress bar
                pbar.update(1)
    
    # Save final results to JSON file
    if results:
        print(f"\nSaving {len(results)} predictions to {output_path}")
        with open(output_path, 'w') as f:
            json.dump({
                'model_path': init_args['model_path'],
                'video_path': video_path,
                'sequence_length': sequence_length,
                'overlap': overlap,
                'classes': list(label_encoder.classes_),
                'results': results
            }, f, indent=2)
        print(f"Results saved to {output_path}")
    else:
        print("No activity predictions were generated!")

# Entry point of the script
if __name__ == '__main__':
    main()