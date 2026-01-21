import torch
import nn.Module
import torch.nn.Functional
from conv_block import conv_block


# kernel_size=3,
#                  filter_strategy="constant",
#                  activation="relu",
#                  dense_neurons=1024):

class imageCNN(nn.Module):
    def __init__(self,input_channels,num_classes,num_blocks=5,base_filters=32,
    kernal_size=3,filter_strategy=constant,activation=relu,dense_neurons=1024):

        super().__init__()
        self.blocks = nn.ModuleList()
        in_ch = input_channels
        out_ch = base_filters
        for i in range(num_blocks):
            self.blocks.append(conv_block
            (in_ch,out_ch,kernal_size,
            use_bn=True,activation=activation,
            pool_kernal=2,dropout=0.2))

            in_ch=out_ch
            if filter_strategy == "doubling":
                out_ch *= 2
            elif filter_strategy == "halving":
                out_ch = max(out_ch // 2, 1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_ch, num_classes)
    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.flatten(x)

    # global average pooling alternative (safer)
    # x = x.mean(dim=(2, 3))

    x = self.fc(x)
    return x

