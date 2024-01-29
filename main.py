import cv2
import numpy as np
import torch
import os
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

# Function to load the MiDaS model
def load_midas_model():
    model = MidasNet("midas.pt", non_negative=True)
    model.eval()
    return model

# Function for MiDaS depth map refinement
def midas_refine_depth(model, depth_map):
    transform = Compose([Resize(
        384, 384,
        resize_target=None,
        keep_aspect_ratio=True,
        ensure_multiple_of=32,
        resize_method="upper_bound",
        image_interpolation_method=cv2.INTER_CUBIC,
    ), NormalizeImage(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), PrepareForNet()])
    
    input_batch = transform({"image": depth_map})["image"]
    input_batch = input_batch.unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_batch)

    output = prediction.squeeze().cpu().numpy()
    output = cv2.resize(output, (depth_map.shape[1], depth_map.shape[0]))
    return output

# Function to generate depth map using OpenCV
def generate_depth_map(left_img_path, right_img_path):
    # Read the images
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    # Initialize StereoSGBM matcher
    window_size = 5
    min_disp = 0
    num_disp = 160 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                   numDisparities = num_disp,
                                   blockSize = window_size,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 100,
                                   speckleRange = 32,
                                   disp12MaxDiff = 1,
                                   P1 = 8*3*window_size**2,
                                   P2 = 32*3*window_size**2)

    # Calculate disparity (depth map)
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

    # Normalize the depth map (just for display)
    depth_map = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return depth_map

# Function to preprocess 3D video and convert it into frames
def preprocess_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Assuming side-by-side 3D format
        height, width, _ = frame.shape
        left_frame = frame[:, :width//2, :]
        right_frame = frame[:, width//2:, :]
        cv2.imwrite(os.path.join(output_folder, f'left_frame_{frame_count:04d}.png'), left_frame)
        cv2.imwrite(os.path.join(output_folder, f'right_frame_{frame_count:04d}.png'), right_frame)
        frame_count += 1
    cap.release()

# Function to save and visualize depth map
def save_and_visualize_depth_map(depth_map, output_path, title='Depth Map'):
    # Normalize for visualization
    normalized_depth = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(output_path, normalized_depth)
    cv2.imshow(title, normalized_depth)
    cv2.waitKey(1)

# Main Processing Function
def process_3d_video(video_path, output_folder):
    # Load MiDaS model
    midas_model = load_midas_model()

    # Preprocess video
    preprocess_video(video_path, output_folder)

    frame_count = 0
    left_frames = sorted([f for f in os.listdir(output_folder) if f.startswith('left_frame_')])
    right_frames = sorted([f for f in os.listdir(output_folder) if f.startswith('right_frame_')])

    for left_frame, right_frame in zip(left_frames, right_frames):
        depth_map = generate_depth_map(os.path.join(output_folder, left_frame),
                                      os.path.join(output_folder, right_frame))
        
        # Save and visualize OpenCV depth map
        save_and_visualize_depth_map(depth_map, os.path.join(output_folder, f'depth_map_opencv_{frame_count:04d}.png'))

        # Refine depth map with MiDaS
        refined_depth = midas_refine_depth(midas_model, depth_map)

        # Save and visualize refined depth map
        save_and_visualize_depth_map(refined_depth, os.path.join(output_folder, f'depth_map_midas_{frame_count:04d}.png'), title='Refined Depth Map')

        frame_count += 1

# Call function
process_3d_video('input.mp4', 'out')