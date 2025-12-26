import torch
import torch.nn as nn
from torchvision import  models
import torch.nn.functional as F

class DenseNetOCR_Attn(nn.Module):
    def __init__(self, num_classes, nhead=8, num_layers=2, hidden_size=256, bidirectional=True,image_height=64, image_width=200):
        super().__init__()
        self.backbone = models.densenet121()
        self.backbone.features.conv0.stride = (2, 1)
        self.backbone.features.pool0.stride = (2, 1)
        self.backbone.features.transition2.pool = nn.AvgPool2d(2, (2, 1))
        #self.backbone.features.transition3.pool = nn.AvgPool2d(2, (2, 1))

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_height, image_width)
            f = self.backbone.features(dummy)
            _, c, h, _ = f.shape
            self.seq_input_size = c * h

        self.map = nn.Linear(self.seq_input_size, hidden_size)
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.cls = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes + 1)

    def forward(self, x):
        f = self.backbone.features(x)
        f = F.relu(f, inplace=True)
        b, c, h, w = f.size()
        f = f.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)  # [B, W, C*H]
        f = self.map(f)  # [B, W, 512]
        f, _ = self.rnn(f)  # [B, W, hidden*2]
        f = self.cls(f)  # [B, W, num_classes+1]
        return f.log_softmax(2)  # [B, W, C]
