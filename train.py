import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import IGazeDataset
import os
from models import FeatureEncoder, TransformerClassifier
import gc
from torch.utils.data import WeightedRandomSampler
from collections import Counter


def train_model(
    feature_encoder,
    transformer_classifier,
    train_loader=None,
    val_loader=None,
    num_classes=106,
    num_epochs=40,
    learning_rate=1e-3,
    weight_decay=0,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    checkpoint_dir=None,
    checkpoint_load=False,
    log_file_dir=None 
):

    def log(message):
        print(message)
        if log_file_dir:
            with open(os.path.join(log_file_dir, 'outputs.txt'), 'a') as f:
                f.write(message + '\n')

    os.makedirs(log_file_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log(f"Device: {device}")
    feature_encoder = feature_encoder.to(device)
    transformer_classifier = transformer_classifier.to(device)

    # Optimizer and Losses
    params = list(feature_encoder.parameters()) + list(transformer_classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_gaze = nn.KLDivLoss(reduction='batchmean')
    start_epoch = 0

    if checkpoint_load:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_epoch27.pt")  # Change manually if needed
        checkpoint = torch.load(checkpoint_path)
        feature_encoder.load_state_dict(checkpoint['feature_encoder_state_dict'])
        transformer_classifier.load_state_dict(checkpoint['transformer_classifier_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        log("Model loaded.")


    # -------- TRAINING --------
    log(f"Resuming from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, num_epochs):
        log('')
        log('')
        log("========== TRAINING STARTED ==========")

        # --- Manual LR decay at epoch 23 and 43 ---
        if epoch in [6, 14, 24, 34, 44]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            new_lr = optimizer.param_groups[0]['lr']
            log(f"[Epoch {epoch+1}] Learning rate manually reduced to {new_lr}")

        feature_encoder.train()
        transformer_classifier.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_gaze_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images,  gaze_gt, labels = batch  # Expect shapes: (B, T, 3, H, W), (B, T, 7, 7), (B,)

            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W).to(device)
            gaze_gt = gaze_gt.to(device)
            labels = labels.to(device)

            # --- Forward ---
            features, attention_maps = feature_encoder(images)  # (B*T, 7, 7, 7)
            features = features.reshape(B, T, -1)  # (B, T, 294)

            # Class prediction
            logits = transformer_classifier(features)  # (B, num_classes)

            # --- Classification loss ---
            loss_cls = criterion_cls(logits, labels)

            # --- Accuracy ---
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # --- Gaze loss ---
            pred_gaze = attention_maps.view(B, T, 7, 7)
            pred_gaze_log = F.log_softmax(pred_gaze.view(B, T, -1), dim=-1)
            gaze_gt_soft = F.softmax(gaze_gt.view(B, T, -1), dim=-1)
            loss_gaze = criterion_gaze(pred_gaze_log, gaze_gt_soft)

            # --- Total loss ---
            loss = loss_cls + loss_gaze

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_gaze_loss += loss_gaze.item()

        avg_loss = total_loss / len(train_loader)
        avg_cls = total_cls_loss / len(train_loader)
        avg_gaze = total_gaze_loss / len(train_loader)
        train_acc = 100.0 * total_correct / total_samples

        log(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Cls: {avg_cls:.4f} | Gaze: {avg_gaze:.4f} | Acc: {train_acc:.2f}%")

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'feature_encoder_state_dict': feature_encoder.state_dict(),
            'transformer_classifier_state_dict': transformer_classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_accuracy': train_acc,
        }, checkpoint_path)
        log(f"Checkpoint saved to {checkpoint_path}")


        # -------- VALIDATION --------

        log("========== VALIDATION STARTED ==========")
        
        feature_encoder.eval()
        transformer_classifier.eval()

        val_loss = 0.0
        val_cls_loss = 0.0
        val_gaze_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = batch[0][0]
                gaze_gt = batch[1][0]
                labels = batch[2][0]  # (B, T, 3, H, W), (B, T, 7, 7), (B,)

                B, T, C, H, W = images.shape
                images = images.view(B * T, C, H, W).to(device)
                gaze_gt = gaze_gt.to(device)
                labels = labels.to(device)

                features, attention_maps = feature_encoder(images)
                features = features.reshape(B, T, -1)
                logits = transformer_classifier(features)

                logits = logits.mean(dim=0)  # Average over time dimension

                # --- Accuracy ---
                preds = torch.argmax(logits)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)

                loss_cls = criterion_cls(logits.unsqueeze(0), labels)
                pred_gaze = attention_maps.view(B, T, 7, 7)
                pred_gaze_log = F.log_softmax(pred_gaze.view(B, T, -1), dim=-1)
                gaze_gt_soft = F.softmax(gaze_gt.view(B, T, -1), dim=-1)
                loss_gaze = criterion_gaze(pred_gaze_log, gaze_gt_soft)

                loss = loss_cls + loss_gaze

                val_loss += loss.item()
                val_cls_loss += loss_cls.item()
                val_gaze_loss += loss_gaze.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_cls = val_cls_loss / len(val_loader)
        avg_val_gaze = val_gaze_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_samples

        log(f"[Epoch {epoch+1}] Val Loss:   {avg_val_loss:.4f} | Cls: {avg_val_cls:.4f} | Gaze: {avg_val_gaze:.4f} | Acc: {val_acc:.2f}%")


    log("Training complete.")


def create_weighted_sampler(labels):
    # Count occurrences of each class
    class_counts = Counter(labels)
    num_samples = len(labels)

    # Compute class weights: inverse of frequency
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

    # Assign a weight to each sample
    sample_weights = [class_weights[label] for label in labels]

    # Create the sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True  # allows re-sampling to balance
    )

    return sampler

if __name__ == "__main__":

    # gc.collect()
    # torch.cuda.empty_cache()

    with open("/workspace/train_split1_filtered.txt", 'r') as f:
        train_labels = [int(line.strip().split()[1]) - 1 for line in f.readlines()]

    
    sampler = create_weighted_sampler(train_labels)


    datapath = '/workspace'

    # Train and validation sets
    train_dataset = IGazeDataset(datapath, 'train', data_split=1)
    val_dataset = IGazeDataset(datapath, 'test', data_split=1)

    train_loader = DataLoader(train_dataset, batch_size=16, pin_memory=True, sampler=sampler, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=10)

    feature_encoder = FeatureEncoder()
    transformer_classifier = TransformerClassifier(num_classes=106)

    train_model(
        feature_encoder=feature_encoder,
        transformer_classifier=transformer_classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=106,
        num_epochs=50,
        learning_rate=1e-4,
        weight_decay=0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_dir='/workspace/checkpoints',
        checkpoint_load=False,
        log_file_dir='/workspace/logs'
    )
