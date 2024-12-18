import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
   

class UNet2D(nn.Module):
    def __init__(self, input_channels, n_classes, features=[64, 128, 256, 512]):
        super(UNet2D, self).__init__()
        
        self.activation = nn.Softmax(dim=1)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Encoder (Downsampling)
        for feature in features:
            self.downs.append(self.conv_block(input_channels, feature))
            input_channels = feature

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.ups.append(self.upconv_block(feature*2, feature))
            self.ups.append(self.conv_block(feature*2, feature))

        # Bottom layer
        self.bottle_neck = self.conv_block(features[-1], features[-1]*2)

        # Final layer
        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottom layer
        x = self.bottle_neck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            # print("x: ", x.shape)
            # print("conn: ", skip_connection.shape)

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        out = self.final_conv(x)
        out = self.activation(out)

        return out

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


def test():
    input_channels = 3
    n_classes = 3

    x = torch.randn((1, input_channels, 161, 161)) # (batch_size, channels, height, width)
    
    model = UNet2D(input_channels, n_classes)
    preds = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Predicted shape: {preds.shape}")
    
    print(summary(model, input_size=(input_channels, 161, 161)))

        


if __name__ == "__main__":
    test()