import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class CaptchaDataset(Dataset):
    def __init__(self, img_dir, label_file=None, transform=None, file_list=None, split=None,base_chars=None,num_base=0):
        self.img_dir = img_dir
        self.transform = transform
        self.split = split
        self.image_files = file_list
        self.base_chars = base_chars
        self.num_base = num_base
        if split != 'test':
            self.df = pd.read_csv(label_file).set_index("filename")

    def __len__(self):
        return len(self.image_files)
    
    def get_label_id(self,char, color_code):
        if char not in self.base_chars:
            return -1
        base_idx = self.base_chars.index(char)
        return base_idx + 1 if color_code == 'r' else base_idx + 1 + self.num_base

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