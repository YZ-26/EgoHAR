import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

def transform(snippet):

    snippet = np.array(snippet)  # (T, H, W, C)
    snippet = torch.from_numpy(snippet).float() / 255.0  # [0,1], shape (T, H, W, C)
    snippet = snippet.permute(0, 3, 1, 2)  # (T, C, H, W)

    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    snippet = (snippet - mean) / std

    return snippet  # shape: (B, C, H, W)

def generate_gaussian_heatmap(xy_list, height, width, sigma):

    T = xy_list.shape[0]

    # Convert normalized coordinates to absolute pixel coordinates
    cx = (xy_list[:, 0] * width).view(T, 1, 1)   # (T, 1, 1)
    cy = (xy_list[:, 1] * height).view(T, 1, 1)  # (T, 1, 1)

    # Create coordinate grid
    xs = torch.arange(width).view(1, 1, width)    # (1, 1, W)
    ys = torch.arange(height).view(1, height, 1)  # (1, H, 1)

    # Compute squared distances
    dx2 = (xs - cx) ** 2
    dy2 = (ys - cy) ** 2

    heatmaps = torch.exp(-(dx2 + dy2) / (2 * sigma ** 2))  # (T, H, W)
    return heatmaps

class IGazeDataset(Dataset):
    def __init__(self, root, mode='train', data_split=1, clip_len=16, sigma=1.5, img_size=(224, 224), max_test_clips=10, test_sparse=True):
        self.root = root
        self.mode = mode
        self.split = data_split
        self.clip_len = clip_len
        self.sigma = sigma
        self.img_size = img_size
        self.max_test = max_test_clips
        self.train_mode = (mode == 'train')
        self.test_sparse = test_sparse

        split_fn = os.path.join(root, f"{mode}_split{data_split}_filtered.txt")
        self.entries = []
        with open(split_fn, 'r') as f:
            for line in f:
                cn, label = line.strip().split()[:2]
                self.entries.append((cn, int(label) - 1))


        self.frame_base = os.path.join(root, 'Frames', f"{mode}{data_split}")
        self.gaze_base = os.path.join(root, 'Gaze_Data', f"{mode}{data_split}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        cn, label = self.entries[idx]
        frame_dir = os.path.join(self.frame_base, cn)
        gaze_path = os.path.join(self.gaze_base, f"{cn}.npy")

        # Parse frame range
        parts = cn.split('-')
        fstart = int(parts[-2][1:])
        fend   = int(parts[-1][1:])
        T = fend - fstart + 1

        # Load gaze data
        gaze_xy = torch.from_numpy(np.load(gaze_path)).float()

        # --- Try to skip 20% on both sides ---
        tentative_skip = int(T * 0.2)
        valid_range = T - 2 * tentative_skip
        skip = tentative_skip if valid_range >= self.clip_len else 0

        if self.train_mode:
            max_start = (T - 2 * skip) - self.clip_len + 1
            offset = np.random.randint(0, max_start) + skip + 1  # +1 for 1-based indexing
            indices = [offset + i for i in range(self.clip_len)]

            snippet = []
            for i in indices:
                frame_path = os.path.join(frame_dir, f"frame_{i:05d}.jpg")
                img = cv2.imread(frame_path)
                if img is None:
                    print(f"[Warning] Skipping missing frame: {frame_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                snippet.append(img)

            imgs = transform(snippet)
            xy = gaze_xy[offset - 1: offset + self.clip_len - 1]
            heatmap = generate_gaussian_heatmap(xy, 7, 7, self.sigma)
            return imgs, heatmap, label

        else:
            img_batch = []
            heatmap_batch = []
            label_batch = []

            effective_range = T - 2 * skip
            jump = self.clip_len
            if self.test_sparse and effective_range > self.clip_len * self.max_test:
                jump = effective_range // self.max_test

            list_start_idx = list(range(skip + 1, skip + 1 + effective_range - self.clip_len + 1, jump))

            for offset in list_start_idx:
                indices = [offset + i for i in range(self.clip_len)]
                snippet = []
                for i in indices:
                    frame_path = os.path.join(frame_dir, f"frame_{i:05d}.jpg")
                    img = cv2.imread(frame_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    snippet.append(img)

                imgs = transform(snippet)
                xy = gaze_xy[offset - 1: offset + self.clip_len - 1]
                heatmap = generate_gaussian_heatmap(xy, 7, 7, self.sigma)
                img_batch.append(imgs)
                heatmap_batch.append(heatmap)
            label_batch.append(label)

            return torch.stack(img_batch), torch.stack(heatmap_batch), torch.tensor(label_batch)

