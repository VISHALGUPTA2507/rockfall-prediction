"""
rockfall_system.py
Single-file AI-based rockfall detection system (training + real-time inference)

Features:
- Train a transfer-learning classifier (ResNet18) on an image dataset with two folders: "rockfall/" and "normal/"
- Save a checkpoint that includes class->index mapping
- Run real-time inference using webcam or video file/stream
  - Uses background subtraction to focus on moving regions and classify ROIs
  - Triggers an alert (visual overlay + saved snapshot + system bell) when rockfall is detected
- Easy CLI: --mode train or --mode run

Usage examples:
  # Train
  python rockfall_system.py --mode train --data_dir dataset --epochs 10 --batch_size 16

  # Run inference on webcam
  python rockfall_system.py --mode run --model_path models/rockfall_resnet18.pt --source 0

  # Run inference on video file
  python rockfall_system.py --mode run --model_path models/rockfall_resnet18.pt --source path/to/video.mp4

Notes & tips:
- Data structure for training (ImageFolder format):
    dataset/
      rockfall/
        r001.jpg
      normal/
        n001.jpg

- If you don't have a webcam: use a recorded video, or stream from your phone (IP Webcam apps) and pass the HTTP/RTSP URL as --source
- This script does NOT remotely access your camera — it runs locally on your machine. You'll need to run it yourself and grant camera access locally.

Dependencies (install with pip):
  pip install torch torchvision opencv-python pillow numpy tqdm

"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

try:
    import cv2
except Exception as e:
    raise RuntimeError("OpenCV is required for inference and real-time video processing. Install opencv-python")

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x):
        return x


# ----------------------------- Config / Utilities -----------------------------

DEFAULTS = {
    'img_size': 224,
    'batch_size': 16,
    'epochs': 8,
    'lr': 1e-3,
    'min_motion_area': 1500,  # pixels
    'alert_threshold': 0.75,   # probability above which to alert
    'alert_cooldown': 6.0,     # seconds between alerts to avoid spam
}


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def create_model(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    # replace final layer
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


# ----------------------------- Training Pipeline -----------------------------

def get_dataloaders(data_dir, img_size=224, batch_size=16):
    data_dir = Path(data_dir)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {
        'train': datasets.ImageFolder(str(data_dir), transform=train_transforms)
    }

    # Split train into train/val (80/20)
    n = len(image_datasets['train'])
    if n == 0:
        raise RuntimeError(f'No images found in {data_dir}. Expect two folders: rockfall/ and normal/')
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_set, val_set = torch.utils.data.random_split(image_datasets['train'], [n_train, n_val])
    # apply val transforms to val_set samples by replacing dataset.transform
    train_set.dataset.transform = train_transforms
    val_set.dataset.transform = val_transforms

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    }

    # class_to_idx from the original ImageFolder (important)
    class_to_idx = image_datasets['train'].class_to_idx
    return dataloaders, class_to_idx


def train_model(data_dir, model_save_path, epochs=8, batch_size=16, img_size=224, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    dataloaders, class_to_idx = get_dataloaders(data_dir, img_size=img_size, batch_size=batch_size)
    num_classes = len(class_to_idx)
    print('Classes found:', class_to_idx)

    model = create_model(num_classes=num_classes, pretrained=True).to(device)

    # Freeze backbone except the final fc
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_acc = 0.0
    ensure_dir(Path(model_save_path).parent)

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        # training
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for inputs, labels in tqdm(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        print(f'Train loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}')

        # validation
        model.eval()
        val_running_corrects = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data).item()
                val_total += inputs.size(0)
        val_acc = val_running_corrects / val_total
        print(f'Val acc: {val_acc:.4f}')

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'model_state': model.state_dict(),
                'class_to_idx': class_to_idx,
                'img_size': img_size,
                'arch': 'resnet18'
            }
            torch.save(checkpoint, model_save_path)
            print('Saved best model to', model_save_path)

    print('Training complete. Best val acc:', best_val_acc)


# ----------------------------- Inference / Real-time -----------------------------


def load_checkpoint(model_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint.get('class_to_idx')
    num_classes = len(class_to_idx)
    img_size = checkpoint.get('img_size', DEFAULTS['img_size'])
    model = create_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    return model, class_to_idx, img_size


def transform_frame_for_model(frame_bgr, img_size):
    # frame_bgr: numpy array in BGR (OpenCV)
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    preprocess = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)  # add batch dim


def alert_actions(frame, bbox, prob, output_dir='alerts'):
    ensure_dir(output_dir)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = Path(output_dir) / f'alert_{ts}.jpg'
    cv2.imwrite(str(fname), frame)
    print(f'ALERT saved snapshot to {fname} prob={prob:.2f}')
    # Try to beep (best-effort)
    try:
        # Windows
        import winsound
        winsound.Beep(1000, 400)
    except Exception:
        try:
            # Mac / Linux: terminal bell fallback
            print('\a')
        except Exception:
            pass


def run_inference(model_path, source=0, img_size=None, min_motion_area=1500, threshold=0.75, cooldown=6.0, output_dir='alerts'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, class_to_idx, ck_img_size = load_checkpoint(model_path, device=device)
    if img_size is None:
        img_size = ck_img_size
    # figure out rockfall class index
    rockfall_idx = None
    for k, v in class_to_idx.items():
        if 'rock' in k.lower() or 'fall' in k.lower():
            rockfall_idx = v
            break
    if rockfall_idx is None:
        # fallback: assume label 1 is rockfall
        rockfall_idx = 1
        print('Warning: Could not find class named "rockfall" — falling back to index 1')
    print('Using rockfall class index:', rockfall_idx)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video source: {source}')

    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

    last_alert_time = 0.0
    frame_count = 0
    ensure_dir(output_dir)
    print('Starting inference loop. Press q to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('End of stream or cannot read frame')
            break
        frame_count += 1
        # resize for speed (keep aspect ratio)
        h, w = frame.shape[:2]
        # motion detection
        fgmask = backSub.apply(frame)
        # morphological clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_any = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_motion_area:
                continue
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            # add padding
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w_box + pad)
            y2 = min(frame.shape[0], y + h_box + pad)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            input_tensor = transform_frame_for_model(roi, img_size).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                rock_prob = float(probs[rockfall_idx]) if rockfall_idx < len(probs) else 0.0

            label = 'NORMAL'
            color = (0, 255, 0)
            if rock_prob >= threshold and (time.time() - last_alert_time) > cooldown:
                label = f'ROCKFALL {rock_prob:.2f}'
                color = (0, 0, 255)
                last_alert_time = time.time()
                alert_actions(frame, (x1, y1, x2, y2), rock_prob, output_dir=output_dir)

            # draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            detected_any = True

        # show mask and frame side-by-side
        mask_rgb = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((cv2.resize(frame, (640, 360)), cv2.resize(mask_rgb, (320, 180))))
        cv2.imshow('Rockfall Detection (press q to quit)', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Stopped.')


# ----------------------------- CLI -----------------------------


def main():
    parser = argparse.ArgumentParser(description='Single-file rockfall detection (train + run)')
    parser.add_argument('--mode', choices=['train', 'run'], required=True)
    parser.add_argument('--data_dir', default='dataset', help='Path to training dataset (ImageFolder format)')
    parser.add_argument('--model_path', default='models/rockfall_resnet18.pt', help='Where to save/load the model')
    parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    parser.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--img_size', type=int, default=DEFAULTS['img_size'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    parser.add_argument('--source', default='0', help='Video source for run mode. 0 for webcam, or path/URL to video stream')
    parser.add_argument('--min_motion_area', type=int, default=DEFAULTS['min_motion_area'])
    parser.add_argument('--threshold', type=float, default=DEFAULTS['alert_threshold'])
    parser.add_argument('--cooldown', type=float, default=DEFAULTS['alert_cooldown'])
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args.data_dir, args.model_path, epochs=args.epochs, batch_size=args.batch_size, img_size=args.img_size, lr=args.lr)
    else:
        # coerce source
        source = args.source
        # if numeric string -> webcam index
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        run_inference(args.model_path, source=source, img_size=args.img_size, min_motion_area=args.min_motion_area, threshold=args.threshold, cooldown=args.cooldown)


if __name__ == '__main__':
    main()
