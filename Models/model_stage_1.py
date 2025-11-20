import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelAttention(nn.Module):
    """Channel attention module for emphasizing important channels"""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on important regions"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.sigmoid(self.conv(x_cat))
        return x * x_out


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class Conv2DBlock(nn.Module):
    """Enhanced 2D convolutional block with optional attention and dilation"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, use_attention=False):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(out_channels)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.use_attention:
            x = self.attention(x)
        return x


class DilatedResidualBlock(nn.Module):
    """
    Residual block with dilated convolutions for capturing long-range dependencies
    Critical for radio propagation modeling where signals travel far from source
    """

    def __init__(self, channels, dilation_rates=[1, 2, 4], use_attention=True):
        super(DilatedResidualBlock, self).__init__()

        # Calculate per-branch channels ensuring they sum to total channels
        num_branches = len(dilation_rates)
        channels_per_branch = channels // num_branches
        remaining_channels = channels % num_branches

        # Multi-scale dilated convolutions with proper channel distribution
        self.dilated_convs = nn.ModuleList()
        for i, rate in enumerate(dilation_rates):
            # Give extra channel to first branches if there's a remainder
            out_ch = channels_per_branch + (1 if i < remaining_channels else 0)
            self.dilated_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_ch, 3, padding=rate, dilation=rate),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )

        # Fusion - input will now exactly match channels
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )

        self.attention = CBAM(channels) if use_attention else None

    def forward(self, x):
        residual = x

        # Apply dilated convolutions at different scales
        multi_scale_features = [conv(x) for conv in self.dilated_convs]
        out = torch.cat(multi_scale_features, dim=1)  # Now sums exactly to 'channels'

        # Fuse and add residual
        out = self.fusion(out)
        out = out + residual
        out = F.relu(out)

        if self.attention is not None:
            out = self.attention(out)
        return out


class ResidualBlock2D(nn.Module):
    """Standard 2D Residual block with attention"""

    def __init__(self, channels, use_attention=True):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = Conv2DBlock(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.attention = CBAM(channels) if use_attention else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        out = F.relu(out)
        if self.attention is not None:
            out = self.attention(out)
        return out


class DownsampleBlock(nn.Module):
    """Downsampling block with residual connection"""

    def __init__(self, in_channels, out_channels, use_attention=True, use_dilated=False):
        super(DownsampleBlock, self).__init__()

        # Main path with strided convolution for downsampling
        if use_dilated:
            residual_block = DilatedResidualBlock(out_channels, use_attention=use_attention)
        else:
            residual_block = ResidualBlock2D(out_channels, use_attention=use_attention)

        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            residual_block
        )

        # Shortcut path for residual connection
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.main_path(x) + self.shortcut(x)


class TransmitterEncoder(nn.Module):
    """
    Specialized encoder for transmitter location
    Encodes the point source into a spatial feature map
    """

    def __init__(self, out_channels=32):
        super(TransmitterEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)


class RadioMapUNet(nn.Module):
    """
    U-Net architecture for 2D radio map construction (Phase 1: 256x256 -> 64x64)

    Key features:
    - Dilated convolutions for long-range radio propagation modeling
    - Multi-scale feature extraction
    - Attention mechanisms for focusing on important regions
    - Separate encoding for transmitter location

    Args:
        in_channels (int): Number of input channels (default: 2, building + transmitter)
        out_channels (int): Number of output channels (default: 1, radio map)
        base_channels (int): Base number of channels in the network (default: 32)
        use_attention (bool): Whether to use attention mechanisms (default: True)
    """

    def __init__(self, out_channels=1, base_channels=32, use_attention=True):
        super(RadioMapUNet, self).__init__()

        # Transmitter location encoder (specialized processing)
        self.tx_encoder = TransmitterEncoder(out_channels=base_channels)

        # Initial building layout processing
        self.building_encoder = nn.Sequential(
            Conv2DBlock(1, base_channels, use_attention=False),
            ResidualBlock2D(base_channels, use_attention=False)
        )

        # Fuse building and transmitter features
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Encoder path
        # 256x256 level
        self.enc1 = nn.Sequential(
            Conv2DBlock(base_channels, base_channels, use_attention=use_attention),
            ResidualBlock2D(base_channels, use_attention=use_attention)
        )

        # 256x256 -> 128x128
        self.down1 = DownsampleBlock(base_channels, base_channels * 2,
                                     use_attention=use_attention, use_dilated=False)
        self.enc2 = ResidualBlock2D(base_channels * 2, use_attention=use_attention)

        # 128x128 -> 64x64
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4,
                                     use_attention=use_attention, use_dilated=True)
        self.enc3 = DilatedResidualBlock(base_channels * 4, use_attention=use_attention)

        # 64x64 -> 32x32 (for deeper feature extraction)
        self.down3 = DownsampleBlock(base_channels * 4, base_channels * 8,
                                     use_attention=use_attention, use_dilated=True)
        self.enc4 = DilatedResidualBlock(base_channels * 8,
                                         dilation_rates=[1, 2, 4, 8],
                                         use_attention=use_attention)

        # Bottleneck at 32x32 with heavy dilated convolutions
        # This captures very long-range dependencies crucial for radio propagation
        self.bottleneck = nn.Sequential(
            DilatedResidualBlock(base_channels * 8, dilation_rates=[1, 2, 4, 8], use_attention=True),
            Conv2DBlock(base_channels * 8, base_channels * 16, use_attention=True),
            DilatedResidualBlock(base_channels * 16, dilation_rates=[1, 4, 8, 16], use_attention=True),
            Conv2DBlock(base_channels * 16, base_channels * 8, use_attention=True),
        )

        # Decoder path
        # 32x32 -> 64x64 (our target resolution)
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4,
                                      kernel_size=2, stride=2)

        # Fusion with skip connection from enc3
        self.dec1 = nn.Sequential(
            Conv2DBlock(base_channels * 8, base_channels * 4, use_attention=use_attention),
            DilatedResidualBlock(base_channels * 4, dilation_rates=[1, 2, 4], use_attention=True),
            ResidualBlock2D(base_channels * 4, use_attention=use_attention)
        )

        # Final refinement at 64x64
        self.final = nn.Sequential(
            Conv2DBlock(base_channels * 4, base_channels * 2, use_attention=True),
            DilatedResidualBlock(base_channels * 2, dilation_rates=[1, 2], use_attention=True),
            Conv2DBlock(base_channels * 2, base_channels, use_attention=False),
            nn.Conv2d(base_channels, out_channels, 1),
        )

    def forward(self, x):
        # Split input channels
        building = x[:, 0:1, :, :]  # Building layout
        transmitter = x[:, 1:2, :, :]  # Transmitter location

        # Separate encoding
        building_feat = self.building_encoder(building)  # 32 x 256 x 256
        tx_feat = self.tx_encoder(transmitter)  # 32 x 256 x 256

        # Fuse features
        fused = self.fusion(torch.cat([building_feat, tx_feat], dim=1))  # 32 x 256 x 256

        # Encoder path
        enc1 = self.enc1(fused)  # 32 x 256 x 256

        enc2 = self.down1(enc1)  # 64 x 128 x 128
        enc2 = self.enc2(enc2)

        enc3 = self.down2(enc2)  # 128 x 64 x 64
        enc3 = self.enc3(enc3)

        enc4 = self.down3(enc3)  # 256 x 32 x 32
        enc4 = self.enc4(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)  # 256 x 32 x 32

        # Decoder
        dec1 = self.up1(bottleneck)  # 128 x 64 x 64
        dec1 = torch.cat([dec1, enc3], dim=1)  # 256 x 64 x 64
        dec1 = self.dec1(dec1)  # 128 x 64 x 64

        # Final output at 64x64
        out = self.final(dec1)  # 1 x 64 x 64

        return out


# Test the model
if __name__ == "__main__":
    # Initialize model
    model = RadioMapUNet(
        out_channels=1,
        base_channels=32,
        use_attention=True
    )

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Example input
    batch_size = 2

    # Building layout (binary: 0=free space, 1=building)
    building_layout = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()

    # Transmitter location (one-hot encoding)
    transmitter = torch.zeros(batch_size, 1, 256, 256)
    # Place transmitter at random location for each sample
    for i in range(batch_size):
        tx_x, tx_y = np.random.randint(0, 256, 2)
        transmitter[i, 0, tx_x, tx_y] = 1.0

    # Combine into input tensor
    input_tensor = torch.cat([building_layout, transmitter], dim=1).to(device)

    # Forward pass
    print("Testing RadioMapUNet (Phase 1: 256x256 -> 64x64):")
    with torch.no_grad():
        output = model(input_tensor)

    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Calculate FLOPs (rough estimate)
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        _ = model(input_tensor)

    print(f"\n  Performance summary:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))