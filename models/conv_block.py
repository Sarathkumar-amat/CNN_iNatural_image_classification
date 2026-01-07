import torch
import torch.nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 use_bn=False, activation="relu",pool_kernal=2,dropout=0.0):
        super().__init__()
        self.conv = nn.conv2d(in_channels=in_channels,out_channels=out_channels,kernal=kernal_size,padding=kernal_size//2)
        self.use_bn=use_bn
        if use_bn:
            self.bn = nn.BatchNorm2D(out_channels)
        self.activation_name = activation.lower()
        self.pool = nn.MaxPool2d(kernal_size=pool_kernal)
        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout2d(dropout)

    def forward(self,x):
        x = self.conv(x)
        if self.use_bn: 
            x = self.bn(cnn_out)
        if self.activation_name == "relu":
            x = F.relu(x)
        elif self.activation_name == "gelu":
            x = F.gelu(x)
        elif self.activation_name == "silu":
            x = F.silu(x)
        elif self.activation_name == "mish":
            x = F.mish(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
        self.pool(x)
        if self.use_dropout:
            x=self.dropout(x)
        return x