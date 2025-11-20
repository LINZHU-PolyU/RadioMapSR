import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelAttention(nn.Module):
    """Channel attention for feature recalibration"""

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
        return x * (avg_out + max_out).view(b, c, 1, 1)


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) for feature extraction
    Uses dense connections for better feature reuse
    """

    def __init__(self, channels, growth_rate=32, num_layers=5):
        super(ResidualDenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels + i * growth_rate, growth_rate, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        # Local feature fusion
        self.lff = nn.Conv2d(channels + num_layers * growth_rate, channels, 1)
        
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        out = self.lff(torch.cat(features, 1))
        return out + x  # Local residual learning


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB)
    Key building block in ESRGAN for powerful feature extraction
    """

    def __init__(self, channels, growth_rate=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_rate)
        self.rdb2 = ResidualDenseBlock(channels, growth_rate)
        self.rdb3 = ResidualDenseBlock(channels, growth_rate)
        
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x  # Residual scaling


class PixelShuffleUpsampler(nn.Module):
    """
    Efficient upsampling using pixel shuffle (sub-pixel convolution)
    Better than transposed convolution for avoiding checkerboard artifacts
    """

    def __init__(self, in_channels, scale_factor=2):
        super(PixelShuffleUpsampler, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.activation(x)
        return x


class ConditionEncoder(nn.Module):
    """
    Encodes conditioning information (building layout + transmitter location)
    This helps the super-resolution network understand the physical constraints
    """

    def __init__(self, in_channels=2, out_channels=64):
        super(ConditionEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
    def forward(self, x):
        return self.encoder(x)


class FusionModule(nn.Module):
    """
    Fuses low-res radio map features with high-res conditioning information
    Uses attention to adaptively weight the conditioning
    """

    def __init__(self, radio_channels, condition_channels, out_channels):
        super(FusionModule, self).__init__()
        self.fusion_conv = nn.Conv2d(
            radio_channels + condition_channels, 
            out_channels, 1
        )
        self.attention = ChannelAttention(out_channels)
        
    def forward(self, radio_features, condition_features):
        fused = torch.cat([radio_features, condition_features], dim=1)
        fused = self.fusion_conv(fused)
        fused = self.attention(fused)
        return fused


class RadioMapSuperResolution(nn.Module):
    """
    Super-Resolution Network for Phase 2: 64x64 -> 256x256 (4x upsampling)
    
    Architecture based on ESRGAN with modifications for radio map reconstruction:
    - Condition-aware processing using building layout and transmitter location
    - RRDB blocks for powerful feature extraction
    - Pixel shuffle for efficient upsampling
    - Multi-scale refinement
    
    Designed for few-shot learning scenarios with limited high-res training data.
    
    Args:
        in_channels (int): Input channels from Phase 1 (default: 1, radio map)
        condition_channels (int): Conditioning channels (default: 2, building + transmitter)
        out_channels (int): Output channels (default: 1, high-res radio map)
        base_channels (int): Base feature channels (default: 64)
        num_rrdb (int): Number of RRDB blocks (default: 16)
        growth_rate (int): Growth rate in RDB (default: 32)
    """

    def __init__(self, in_channels=1, condition_channels=2, out_channels=1, 
                 base_channels=64, num_rrdb=16, growth_rate=32):
        super(RadioMapSuperResolution, self).__init__()
        
        # Shallow feature extraction from low-res radio map
        self.shallow_feature = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Condition encoder (processes building layout + transmitter at target resolution)
        self.condition_encoder = ConditionEncoder(condition_channels, base_channels)
        
        # Fusion module at input resolution (64x64)
        self.input_fusion = FusionModule(base_channels, base_channels, base_channels)
        
        # Deep feature extraction with RRDB blocks
        self.rrdb_blocks = nn.ModuleList([
            RRDB(base_channels, growth_rate) for _ in range(num_rrdb)
        ])
        
        # Deep feature fusion
        self.deep_fusion = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        
        # Upsampling path: 64x64 -> 128x128 -> 256x256
        self.upsample1 = PixelShuffleUpsampler(base_channels, scale_factor=2)  # 128x128
        self.upsample2 = PixelShuffleUpsampler(base_channels, scale_factor=2)  # 256x256
        
        # Condition fusion at intermediate resolution (128x128)
        self.mid_fusion = FusionModule(base_channels, base_channels, base_channels)
        
        # Condition fusion at final resolution (256x256)
        self.final_fusion = FusionModule(base_channels, base_channels, base_channels)
        
        # High-resolution refinement
        self.hr_refine = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, out_channels, 3, padding=1)
        )
        
        # For few-shot learning: learnable scaling parameter
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, low_res_radiomap, condition_input):
        """
        Args:
            low_res_radiomap: Low-resolution radio map from Phase 1, shape (B, 1, 64, 64)
            condition_input: High-res building layout + transmitter, shape (B, 2, 256, 256)
        
        Returns:
            High-resolution radio map, shape (B, 1, 256, 256)
        """
        
        # Extract shallow features from low-res radio map
        shallow_feat = self.shallow_feature(low_res_radiomap)  # (B, 64, 64, 64)
        
        # Process conditioning information at different scales
        condition_hr = self.condition_encoder(condition_input)  # (B, 64, 256, 256)
        condition_lr = F.adaptive_avg_pool2d(condition_hr, (64, 64))  # (B, 64, 64, 64)
        condition_mid = F.adaptive_avg_pool2d(condition_hr, (128, 128))  # (B, 64, 128, 128)
        
        # Fuse low-res radio features with conditioning
        fused_feat = self.input_fusion(shallow_feat, condition_lr)
        
        # Deep feature extraction with RRDB blocks
        deep_feat = fused_feat
        for rrdb in self.rrdb_blocks:
            deep_feat = rrdb(deep_feat)
        
        # Global residual learning
        deep_feat = self.deep_fusion(deep_feat)
        deep_feat = deep_feat + shallow_feat
        
        # Upsampling path with multi-scale conditioning
        # 64x64 -> 128x128
        up_feat = self.upsample1(deep_feat)  # (B, 64, 128, 128)
        up_feat = self.mid_fusion(up_feat, condition_mid)
        
        # 128x128 -> 256x256
        up_feat = self.upsample2(up_feat)  # (B, 64, 256, 256)
        up_feat = self.final_fusion(up_feat, condition_hr)
        
        # High-resolution refinement
        output = self.hr_refine(up_feat)
        
        # Apply learnable scaling (helps in few-shot scenarios)
        output = output * self.output_scale
        
        return output


class RadioMapSuperResolutionWithDiscriminator(nn.Module):
    """
    Complete GAN-based super-resolution system with discriminator
    For few-shot learning with perceptual quality
    """
    
    def __init__(self, generator_params=None):
        super(RadioMapSuperResolutionWithDiscriminator, self).__init__()
        
        if generator_params is None:
            generator_params = {}
        
        self.generator = RadioMapSuperResolution(**generator_params)
        self.discriminator = PatchDiscriminator(in_channels=1)
        
    def forward(self, low_res_radiomap, condition_input):
        return self.generator(low_res_radiomap, condition_input)


class PatchDiscriminator(nn.Module):
    """
    Patch-based discriminator for GAN training
    Helps generate realistic textures in few-shot scenarios
    """

    def __init__(self, in_channels=1, base_channels=64):
        super(PatchDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 16x16
            nn.Conv2d(base_channels * 8, 1, 4, padding=1)
        )
        
    def forward(self, x):
        return self.model(x)


# Test the models
if __name__ == "__main__":
    print("="*80)
    print("Testing Phase 2: Super-Resolution Network (64x64 -> 256x256)")
    print("="*80)
    
    # Initialize model
    model = RadioMapSuperResolution(
        in_channels=1,
        condition_channels=2,
        out_channels=1,
        base_channels=64,
        num_rrdb=16,
        growth_rate=32
    )

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Example inputs
    batch_size = 2
    
    # Low-resolution radio map from Phase 1
    low_res_radiomap = torch.randn(batch_size, 1, 64, 64).to(device)
    
    # High-resolution conditioning input (building + transmitter at 256x256)
    building_layout_hr = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    transmitter_hr = torch.zeros(batch_size, 1, 256, 256)
    for i in range(batch_size):
        tx_x, tx_y = np.random.randint(0, 256, 2)
        transmitter_hr[i, 0, tx_x, tx_y] = 1.0
    
    condition_input = torch.cat([building_layout_hr, transmitter_hr], dim=1).to(device)
    
    # Forward pass
    print("\nGenerator Forward Pass:")
    with torch.no_grad():
        output = model(low_res_radiomap, condition_input)
    
    print(f"  Low-res input shape: {low_res_radiomap.shape}")
    print(f"  Condition input shape: {condition_input.shape}")
    print(f"  High-res output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Generator parameters:")
    print(f"    Total: {total_params:,}")
    print(f"    Trainable: {trainable_params:,}")
    
    # Test discriminator
    print("\n" + "="*80)
    print("Testing Discriminator:")
    print("="*80)
    
    discriminator = PatchDiscriminator(in_channels=1).to(device)
    
    with torch.no_grad():
        disc_output = discriminator(output)
    
    print(f"  Discriminator output shape: {disc_output.shape}")
    print(f"  Output range: [{disc_output.min().item():.4f}, {disc_output.max().item():.4f}]")
    
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"  Discriminator parameters: {disc_params:,}")
    
    print("\n" + "="*80)
    print("Model initialization complete!")
    print("="*80)
