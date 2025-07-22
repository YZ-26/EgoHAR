import os
import cv2
from tqdm import tqdm
import numpy as np
import re


def check_missing_clips(base_video_dir, train_split_path, test_split_path):
    existing_videos = set()
    for root, _, files in os.walk(base_video_dir):
        for file in files:
            if file.endswith(".mp4"):
                clip_name = file.replace(".mp4", "")
                existing_videos.add(clip_name)

    print(f"Total existing video clips: {len(existing_videos)}")

    def check_missing(split_file):
        with open(split_file, "r") as f:
            lines = f.readlines()

        missing = []
        for line in lines:
            clip_name = line.strip().split()[0]
            if clip_name not in existing_videos:
                missing.append(clip_name)
        return missing

    missing_train = check_missing(train_split_path)
    missing_test = check_missing(test_split_path)

    print(f"\nMissing in train split: {len(missing_train)}")
    for clip in missing_train:
        print("❌", clip)

    print(f"\nMissing in test split: {len(missing_test)}")
    for clip in missing_test:
        print("❌", clip)

def get_existing_video_map(base_video_dir):
    video_map = {}  # key: clip name, value: full path
    for root, _, files in os.walk(base_video_dir):
        for file in files:
            if file.endswith(".mp4"):
                clip_name = file.replace(".mp4", "")
                video_map[clip_name] = os.path.join(root, file)
    return video_map

def read_split(split_path):
    with open(split_path, "r") as f:
        return [line.strip().split()[0] for line in f]

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    frame_id = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_id:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_id += 1
    cap.release()

def process_split(split_name, split_path, video_map, output_base_dir):
    clip_names = read_split(split_path)
    missing = []
    skipped = []

    for clip_name in tqdm(clip_names):
        if clip_name not in video_map:
            missing.append(clip_name)
            continue

        video_path = video_map[clip_name]
        output_dir = os.path.join(output_base_dir, split_name, clip_name)

        if os.path.exists(output_dir):
            skipped.append(clip_name)
            continue

        extract_frames(video_path, output_dir)

    print(f"\n✅ Finished processing '{split_name}' split.")
    print(f"✔️  Extracted: {len(clip_names) - len(missing) - len(skipped)}")
    print(f"⏭️  Skipped (already exist): {len(skipped)}")

    if missing:
        print(f"❌ Missing {len(missing)} clips:")
        for m in missing:
            print("   -", m)



def count_frames_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return -1
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def count_images_in_folder(folder_path, extensions=('.jpg', '.png')):
    return len([f for f in os.listdir(folder_path) if f.endswith(extensions)])

def check_frame_mismatch(root_frame_dir, video_map):
    mismatch = []
    for video_name in tqdm(os.listdir(root_frame_dir)):
        video_frame_count = count_frames_in_video(video_map[video_name])
        frame_folder = os.path.join(root_frame_dir, video_name)
        if not os.path.exists(frame_folder):
            print(f"Missing folder for {video_name}: {frame_folder}")
            continue
        image_count = count_images_in_folder(frame_folder)

        if video_frame_count != image_count:
            mismatch.append(video_name)
    return mismatch



if __name__ == "__main__":
    # Configuration paths
    base_video_dir = "/workspace/data/EGTEA/Trimmed_Action_Clips/cropped_clips"
    train_split_path = "/workspace/EgoHAR/train_split1_filtered.txt"
    test_split_path = "/workspace/EgoHAR/test_split1_filtered.txt"
    output_base_dir = "/workspace/Frames"
    


    # Check missing clips in splits
    print("\n=== Checking missing clips ===")
    check_missing_clips(base_video_dir, train_split_path, test_split_path)

    # Process video frames
    print("\n=== Processing video frames ===")
    video_map = get_existing_video_map(base_video_dir)
    print(f"Total videos found: {len(video_map)}")

    process_split("train1", train_split_path, video_map, output_base_dir)
    process_split("test1", test_split_path, video_map, output_base_dir)

    # Check frame folder stats
    folder_path = '/workspace/Frames/train1'
    if os.path.exists(folder_path):
        num_files = len(os.listdir(folder_path))
        print(f"\nNumber of folders directly in train1': {num_files}")
    
    folder_path = '/workspace/Frames/test1'
    if os.path.exists(folder_path):
        num_files = len(os.listdir(folder_path))
        print(f"\nNumber of folders directly in test1: {num_files}")


    # Check frame count mismatches
    root_frame_dir = "/workspace/Frames/train1"
    if os.path.exists(root_frame_dir):
        mismatch = check_frame_mismatch(root_frame_dir, video_map)
        print("\nVideos with frame count mismatches in train1:")
        print(mismatch)

    root_frame_dir = "/workspace/Frames/test1"
    if os.path.exists(root_frame_dir):
        mismatch = check_frame_mismatch(root_frame_dir, video_map)
        print("\nVideos with frame count mismatches in test1:")
        print(mismatch)
