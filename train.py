import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import IGazeDataset
import os
from models import FeatureEncoder, TransformerClassifier

def train_model(
    feature_encoder,
    transformer_classifier,
    train_loader=None,
    val_loader=None,
    num_classes=106,
    num_epochs=10,
    learning_rate=1e-4,
    weight_decay=0,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    checkpoint_dir=None,
    checkpoint_load=False
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Device: {device}")
    feature_encoder = feature_encoder.to(device)
    transformer_classifier = transformer_classifier.to(device)

    # Optimizer and Losses
    params = list(feature_encoder.parameters()) + list(transformer_classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_gaze = nn.KLDivLoss(reduction='batchmean')
    start_epoch = 0

    if checkpoint_load:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_epoch9.pt")  # Change manually if needed
        checkpoint = torch.load(checkpoint_path)
        feature_encoder.load_state_dict(checkpoint['feature_encoder_state_dict'])
        transformer_classifier.load_state_dict(checkpoint['transformer_classifier_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print("Model loaded.")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=1,
        threshold=0.2,
        threshold_mode='abs'
    )


    # -------- TRAINING --------
    print("========== TRAINING STARTED ==========")
    print(f"Resuming from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, num_epochs):
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
            loss = loss_cls + 0.5 * loss_gaze

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

        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Cls: {avg_cls:.4f} | Gaze: {avg_gaze:.4f} | Acc: {train_acc:.2f}%")

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'feature_encoder_state_dict': feature_encoder.state_dict(),
            'transformer_classifier_state_dict': transformer_classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_accuracy': train_acc,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


        # -------- VALIDATION --------

        print("========== VALIDATION STARTED ==========")
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

                logits = logits.mean(dim=0, keepdim=True)  # Average over time dimension

                # --- Accuracy ---
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)

                loss_cls = criterion_cls(logits, labels)
                pred_gaze = attention_maps.view(B, T, 7, 7)
                pred_gaze_log = F.log_softmax(pred_gaze.view(B, T, -1), dim=-1)
                gaze_gt_soft = F.softmax(gaze_gt.view(B, T, -1), dim=-1)
                loss_gaze = criterion_gaze(pred_gaze_log, gaze_gt_soft)

                loss = loss_cls + 0.5 * loss_gaze

                val_loss += loss.item()
                val_cls_loss += loss_cls.item()
                val_gaze_loss += loss_gaze.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_cls = val_cls_loss / len(val_loader)
        avg_val_gaze = val_gaze_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_samples
        scheduler.step(avg_val_loss)
        print("Current LR:", scheduler.optimizer.param_groups[0]['lr'])

        print(f"[Epoch {epoch+1}] Val Loss:   {avg_val_loss:.4f} | Cls: {avg_val_cls:.4f} | Gaze: {avg_val_gaze:.4f} | Acc: {val_acc:.2f}%")


    print("Training complete.")


if __name__ == "__main__":

    datapath = '/workspace'

    # Train and validation sets
    train_dataset = IGazeDataset(datapath, 'train', data_split=1)
    val_dataset = IGazeDataset(datapath, 'test', data_split=1)

    train_loader = DataLoader(train_dataset, batch_size=8, pin_memory=True, num_workers=os.cpu_count())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count())

    feature_encoder = FeatureEncoder()
    transformer_classifier = TransformerClassifier(num_classes=106)

    train_model(
        feature_encoder=feature_encoder,
        transformer_classifier=transformer_classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=106,
        num_epochs=10,
        learning_rate=1e-4,
        weight_decay=0.01,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_dir='/workspace/checkpoints',
        checkpoint_load=False
    )
