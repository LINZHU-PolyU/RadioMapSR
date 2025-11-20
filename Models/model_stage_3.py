import os
import torch
import torch.nn as nn
from Models.model_stage_1 import RadioMapUNet
from Models.model_stage_2 import RadioMapSuperResolution


class TwoStageModel(nn.Module):
    """
    Combined two-stage model for end-to-end training
    Stage 1: Building + Transmitter -> Low-res radio map
    Stage 2: Building + Transmitter + Low-res radio map -> High-res radio map
    """

    def __init__(self, stage1_model=None, stage2_model=None,
                 base_channels_stage1=32, use_attention_stage1=True,
                 base_channels_stage2=32, num_rrdb=16, growth_rate=32,
                 freeze_stage1=False, freeze_stage2=False):
        super(TwoStageModel, self).__init__()

        # Initialize or use provided Stage 1 model
        if stage1_model is not None:
            self.stage1 = stage1_model
        else:
            self.stage1 = RadioMapUNet(
                out_channels=1,
                base_channels=base_channels_stage1,
                use_attention=use_attention_stage1
            )

        # Initialize or use provided Stage 2 model
        if stage2_model is not None:
            self.stage2 = stage2_model
        else:
            self.stage2 = RadioMapSuperResolution(
                in_channels=1,
                condition_channels=2,
                out_channels=1,
                base_channels=base_channels_stage2,
                num_rrdb=num_rrdb,
                growth_rate=growth_rate,
            )

        # Optionally freeze models
        if freeze_stage1:
            self.freeze_stage1()
        if freeze_stage2:
            self.freeze_stage2()

    def freeze_stage1(self):
        """Freeze Stage 1 parameters"""
        for param in self.stage1.parameters():
            param.requires_grad = False
        print("Stage 1 model frozen")

    def unfreeze_stage1(self):
        """Unfreeze Stage 1 parameters"""
        for param in self.stage1.parameters():
            param.requires_grad = True
        print("Stage 1 model unfrozen")

    def freeze_stage2(self):
        """Freeze Stage 2 parameters"""
        for param in self.stage2.parameters():
            param.requires_grad = False
        print("Stage 2 model frozen")

    def unfreeze_stage2(self):
        """Unfreeze Stage 2 parameters"""
        for param in self.stage2.parameters():
            param.requires_grad = True
        print("Stage 2 model unfrozen")

    def forward(self, inputs, return_intermediate=True):
        """
        Forward pass through both stages
        """
        # Stage 1: Building + Transmitter -> Low-res radio map
        low_res_output = self.stage1(inputs)

        # Stage 2: Building + Transmitter + Low-res radio map -> High-res radio map
        high_res_output = self.stage2(low_res_output, inputs)

        if return_intermediate:
            return low_res_output, high_res_output
        else:
            return high_res_output

    def load_pretrained_weights(self, stage1_checkpoint_path=None, stage2_checkpoint_path=None, device='cuda'):
        """Load pretrained weights for both stages"""
        loaded_stages = []

        if stage1_checkpoint_path and os.path.exists(stage1_checkpoint_path):
            checkpoint = torch.load(stage1_checkpoint_path, map_location=device, weights_only=False)
            self.stage1.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 0)
            print(f"Loaded Stage 1 weights from epoch {epoch + 1}")
            loaded_stages.append(1)

        if stage2_checkpoint_path and os.path.exists(stage2_checkpoint_path):
            checkpoint = torch.load(stage2_checkpoint_path, map_location=device, weights_only=False)
            self.stage2.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 0)
            print(f"Loaded Stage 2 weights from epoch {epoch + 1}")
            loaded_stages.append(2)

        return loaded_stages