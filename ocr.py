import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from PIL import Image
import os
import pandas as pd
import numpy as np
import csv
import random
import math
import argparse
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
num_base = len(base_chars)  
image_height = 64
image_width = 200

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',default=64, help='Batch size')
    parser.add_argument('--epochs', default=50, help='Training epochs')
    parser.add_argument('--warmup_epochs', default=5, help='Warmup epochs')
    parser.add_argument('--output_dir', default="./output/", help='Directory to save outputs')
    parser.add_argument('--train_img_dir', default="./train/images/", help='Training images')
    parser.add_argument('--train_label', default="./train/labels.csv", help='Training labels')
    parser.add_argument('--test_img_dir', default="./test/images/", help='Test images')
    parser.add_argument('--submission_file', default="submission.csv", help='Submission file')
    return parser.parse_args()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class DenseNetOCR_Attn(nn.Module):
    def __init__(self, num_classes, nhead=8, num_layers=2):
        super().__init__()
        self.backbone = models.densenet121()
        self.backbone.features.conv0.stride = (2, 1)
        self.backbone.features.pool0.stride = (2, 1)
        self.backbone.features.transition2.pool = nn.AvgPool2d(2, (2, 1))

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_height, image_width)
            f = self.backbone.features(dummy)
            _, c, h, _ = f.shape
            self.seq_input_size = c * h

        self.map = nn.Linear(self.seq_input_size, 512)
        self.pos_encoder = PositionalEncoding(d_model=512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=nhead, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.attn_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls = nn.Linear(512, num_classes + 1)

    def forward(self, x):
        f = self.backbone.features(x)
        f = F.relu(f, inplace=True)
        b, c, h, w = f.size()
        f = f.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)  # [B, W, C*H]
        f = self.map(f)  # [B, W, 512]
        f = self.pos_encoder(f)  # [B, W, 512]
        f = self.attn_encoder(f)  # [B, W, 512]
        f = self.cls(f)  # [B, W, num_classes+1]
        return f.log_softmax(2)  # [B, W, C]


class CaptchaDataset(Dataset):
    def __init__(self, img_dir, label_file=None, transform=None, file_list=None, split=None):
        self.img_dir = img_dir
        self.transform = transform
        self.split = split
        self.image_files = file_list

        if split != 'test':
            self.df = pd.read_csv(label_file).set_index("filename")

    def __len__(self):
        return len(self.image_files)
    
    def get_label_id(self,char, color_code):
        if char not in base_chars:
            return -1
        base_idx = base_chars.index(char)
        return base_idx + 1 if color_code == 'r' else base_idx + 1 + num_base

    def __getitem__(self, idx):
        name = self.image_files[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.split == "test":
            return img, name

        row = self.df.loc[name]
        chars = str(row["all_label"]).upper()
        colors = str(row["color"]).lower()

        target = []
        for c, col in zip(chars, colors):
            lid =  self.get_label_id(c, col)
            if lid != -1:
                target.append(lid)

        return img, torch.tensor(target, dtype=torch.long), torch.tensor(len(target), dtype=torch.long)


def collate_fn(batch):
    imgs, targets, lens = zip(*batch)
    return torch.stack(imgs), torch.cat(targets), torch.stack(lens)


def decode_output(sequence):
    res = ""
    last = 0
    for p in sequence:
        if p != last and p != 0 and p <= num_base:
            res += base_chars[p - 1]
        last = p
    return res
    

def train(args):
    all_files = sorted(os.listdir(args.train_img_dir))
    random.seed(42)
    random.shuffle(all_files)
    split = int(0.95 * len(all_files))
    train_files = all_files[:split]
    val_files = all_files[split:]
    os.makedirs(args.output_dir, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dataset = CaptchaDataset(args.train_img_dir, args.train_label, transform_train, train_files, "train")
    val_dataset = CaptchaDataset(args.train_img_dir, args.train_label, transform_val, val_files, "val")

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    model = DenseNetOCR_Attn(num_base * 2).to(device)
    
    model_dict = torch.load("./densenet121.pth", map_location='cpu')
    new_state_dict = {}
    for k, v in model_dict.items():
        new_k = k
        new_k = new_k.replace('norm.1', 'norm1')
        new_k = new_k.replace('norm.2', 'norm2')
        new_k = new_k.replace('conv.1', 'conv1')
        new_k = new_k.replace('conv.2', 'conv2')
        new_state_dict[new_k] = v
    model.backbone.load_state_dict(new_state_dict)
    model = model.to(device)

    opt = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 3e-4},
        {"params": model.map.parameters(), "lr": 1e-3},
        {"params": model.attn_encoder.parameters(), "lr": 3e-4},
        {"params": model.cls.parameters(), "lr": 2e-3},
    ], weight_decay=1e-4)

    scheduler = SequentialLR(
        opt,
        [
            LinearLR(opt, start_factor=0.1, total_iters=args.warmup_epochs),
            CosineAnnealingLR(opt, T_max=args.epochs - args.warmup_epochs, eta_min=1e-5)
        ],
        milestones=[args.warmup_epochs]
    )
    crit = nn.CTCLoss(blank=0, zero_infinity=True)

    writer = SummaryWriter(log_dir=f"{args.output_dir}/logs")
    best = 0
    steps = 0
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, (imgs, tgt, lens) in enumerate(train_dataloader):
            imgs, tgt, lens = imgs.to(device), tgt.to(device), lens.to(device)
            out = model(imgs)
            out = out.permute(1, 0, 2)

            inp_len = torch.full((imgs.size(0),), out.size(0), device=device, dtype=torch.long)
            loss = crit(out, tgt, inp_len, lens)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_train_loss += loss.item()
            if batch_idx % 10 == 0:
                writer.add_scalar('Train/Loss', loss.item(), steps)
            steps += 1

        avg_train_loss = total_train_loss / len(train_dataloader)
        model.eval()
        correct = total = 0
        val_loss_sum = 0
        with torch.no_grad():
            for batch_idx,(imgs, tgt, lens) in enumerate(val_dataloader):
                imgs, tgt, lens = imgs.to(device), tgt.to(device), lens.to(device)
                out = model(imgs)
                out_permute = out.permute(1, 0, 2)
                inp_len = torch.full((imgs.size(0),), out_permute.size(0), device=device, dtype=torch.long)
                
                v_loss = crit(out_permute, tgt, inp_len, lens)
                val_loss_sum += v_loss.item()

                preds = out.argmax(2).cpu().numpy()
                start = 0
                for i in range(imgs.size(0)):
                    true_ids = tgt[start:start+lens[i]]
                    start += lens[i]

                    true_red = "".join(base_chars[t-1] for t in true_ids if t <= num_base)
                    pred_red = decode_output(preds[i])

                    if true_red == pred_red:
                        correct += 1
                    total += 1

        avg_val_loss = val_loss_sum / len(val_dataloader)
        acc = correct / total
        print(f"Epoch {epoch+1}/{args.epochs} | Lr: {opt.param_groups[0]['lr']:.6f} | Train Loss: {avg_train_loss:.4f} | Val Acc: {acc:.2%}",flush=True)
        writer.add_scalar('Train/LR', opt.param_groups[0]['lr'], epoch)
        writer.add_scalar('Val/Accuracy', acc, epoch)
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_epoch.pth"))
            print("Saved best checkpoint.", flush=True)

        torch.save(model.state_dict(), os.path.join(args.output_dir, "last.pth"))
        scheduler.step()
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"{epoch+1},Lr: {opt.param_groups[0]['lr']:.6f},Train loss: {avg_train_loss:.4f},Val acc{acc:.4f}\n")

    print("Best Val Acc:", best)


def predict(args):
    model = DenseNetOCR_Attn(num_base * 2).to(device)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_epoch.pth")))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    files = sorted(os.listdir(args.test_img_dir))
    ds = CaptchaDataset(args.test_img_dir, transform=transform, file_list=files, split="test")
    dl = DataLoader(ds, args.batch_size, shuffle=False)

    res = []
    with torch.no_grad():
        for imgs, names in dl:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(2).cpu().numpy()
            for p, n in zip(preds, names):
                res.append([n, decode_output(p)])

    with open(os.path.join(args.output_dir, args.submission_file), "w", newline="") as f:
        csv.writer(f).writerows([["id","label"]] + res)

if __name__ == "__main__":
    args = args_parser()
    train(args)
    predict(args)
