import torch
import torch.nn as nn
from torchvision import  models
import torch.nn.functional as F
from util import PositionalEncoding

class DenseNetOCR_Attn(nn.Module):
    def __init__(self, num_classes, nhead=8, num_layers=2,img_height=64, img_width=200):
        super().__init__()
        self.backbone = models.densenet121()
        self.backbone.features.conv0.stride = (2, 1)
        self.backbone.features.pool0.stride = (2, 1)
        self.backbone.features.transition2.pool = nn.AvgPool2d(2, (2, 1))

        # 动态计算 feature map 尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_height, img_width)
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