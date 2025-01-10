import torch
import torch.nn as nn
import torch.nn.functional as F

class HCNN(nn.Module):
    def __init__(self, num_of_conv_layers=3, kernel_size=3, normalisation="batch", dropout=False, skipping=False, dropout_fc=0.5, num_groups=8):
        super().__init__()

        def get_norm_layer(channels, norm_type):
            if norm_type == "batch":
                return nn.BatchNorm2d(channels)
            elif norm_type == "group":
                return nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels)
            elif norm_type == "layer":
                class DynamicLayerNorm(nn.Module):
                    def __init__(self, channels):
                        super().__init__()
                        self.norm = None
                        self.channels = channels
                        
                    def forward(self, x):
                        if self.norm is None or self.norm.normalized_shape != [self.channels, x.size(2), x.size(3)]:
                            self.norm = nn.LayerNorm([self.channels, x.size(2), x.size(3)]).to(x.device)
                        return self.norm(x)
                return DynamicLayerNorm(channels)
            elif norm_type == "instance":
                return nn.InstanceNorm2d(channels, affine=True)
            elif norm_type == "Adain":
                class AdaIN(nn.Module):
                    def __init__(self, channels):
                        super().__init__()
                        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
                        self.style_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
                        self.style_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
                        
                    def forward(self, x):
                        x = self.instance_norm(x)
                        return x * self.style_scale + self.style_bias
                
                return AdaIN(channels)
            else:    
                return nn.Identity()
                
        # First convolutional block with separate components for skip connections
        self.conv1_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=kernel_size, padding="same")
        self.conv1_norm = get_norm_layer(32, normalisation)
        self.conv1_relu = nn.ReLU()
        self.conv1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block with separate components
        self.conv2_conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding="same")
        self.conv2_norm = get_norm_layer(64, normalisation)
        self.conv2_relu = nn.ReLU()
        self.conv2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block with separate components
        self.conv3_conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding="same")
        self.conv3_norm = get_norm_layer(128, normalisation)
        self.conv3_relu = nn.ReLU()
        self.conv3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Repeated conv block with separate components for skip connections
        self.conv_rep_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same"),
                get_norm_layer(32, normalisation),
                nn.ReLU()
            ) for _ in range(num_of_conv_layers-3)
        ])

        # 1x1 convolutions for skip connections between different channel sizes
        self.skip1_proj = nn.Conv2d(32, 64, kernel_size=1) if skipping else None
        self.skip2_proj = nn.Conv2d(64, 128, kernel_size=1) if skipping else None
        
        # Fully connected layers with reduced complexity
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_fc) if dropout else nn.Identity(),
            nn.Linear(256, 10)
        )
        
        self.num_of_conv_layers = num_of_conv_layers
        self.skipping = skipping
        
    def forward(self, x):
        # First block with potential skip for repeated conv layers
        x = self.conv1_conv(x)
        x = self.conv1_norm(x)
        x = self.conv1_relu(x)
        x = self.conv1_pool(x)
        
        # Handle repeated convolutions with skip connections
        if self.skipping:
            
            for conv_block in self.conv_rep_blocks:
                identity = x
                x = conv_block(x)
                x = x + identity  # Skip connection
            
        else:
            for conv_block in self.conv_rep_blocks:
                x = conv_block(x)
        
        # Second block with skip connection
        identity = x
        x = self.conv2_conv(x)
        x = self.conv2_norm(x)
        x = self.conv2_relu(x)
        if self.skipping:
            # Project identity to match channels
            identity = self.skip1_proj(identity)
            x = x + identity
        x = self.conv2_pool(x)
        
        # Third block with skip connection
        identity = x
        x = self.conv3_conv(x)
        x = self.conv3_norm(x)
        x = self.conv3_relu(x)
        if self.skipping:
            # Project identity to match channels
            identity = self.skip2_proj(identity)
            x = x + identity
        x = self.conv3_pool(x)
        
        x = self.fc(x)
        return x


class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Initialize the low-rank matrices
        self.U = nn.Parameter(torch.Tensor(out_features, rank))
        self.S = nn.Parameter(torch.Tensor(rank))
        self.V = nn.Parameter(torch.Tensor(rank, in_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize using normal distribution
        nn.init.kaiming_normal_(self.U)
        nn.init.kaiming_normal_(self.V)
        nn.init.ones_(self.S)
        
    def forward(self, x):
        # Compute W = U * S * V^T
        return torch.mm(torch.mm(x, self.V.t()) * self.S, self.U.t())
    
    @staticmethod
    def from_dense(linear_layer, rank):
        """Convert a dense linear layer to SVD format using truncated SVD"""
        device = linear_layer.weight.device
        U, S, V = torch.svd(linear_layer.weight.data)
        
        # Create new SVD layer
        svd_layer = SVDLinear(linear_layer.in_features, 
                             linear_layer.out_features, 
                             rank)
        
        # Set truncated parameters
        svd_layer.U.data = U[:, :rank].to(device)
        svd_layer.S.data = S[:rank].to(device)
        svd_layer.V.data = V[:, :rank].t().to(device)
        
        return svd_layer

class CompressedHCNN(HCNN):
    def __init__(self, base_model, compression_ratio):
        """
        Initialize compressed model from base HCNN model
        compression_ratio: float between 0 and 1 for compression level
        """
        super().__init__()
        
        # Copy convolutional layers from base model
        self.conv1_conv = base_model.conv1_conv
        self.conv1_norm = base_model.conv1_norm
        self.conv1_relu = base_model.conv1_relu
        self.conv1_pool = base_model.conv1_pool
        
        self.conv2_conv = base_model.conv2_conv
        self.conv2_norm = base_model.conv2_norm
        self.conv2_relu = base_model.conv2_relu
        self.conv2_pool = base_model.conv2_pool
        
        self.conv3_conv = base_model.conv3_conv
        self.conv3_norm = base_model.conv3_norm
        self.conv3_relu = base_model.conv3_relu
        self.conv3_pool = base_model.conv3_pool
        
        self.conv_rep_blocks = base_model.conv_rep_blocks
        self.skip1_proj = base_model.skip1_proj
        self.skip2_proj = base_model.skip2_proj
        
        # Replace FC layers with compressed versions
        fc_layers = []
        for module in base_model.fc:
            if isinstance(module, nn.Linear):
                min_dim = min(module.in_features, module.out_features)
                rank = max(1, int(min_dim * compression_ratio))
                fc_layers.append(SVDLinear.from_dense(module, rank))
            else:
                fc_layers.append(module)
        
        self.fc = nn.Sequential(*fc_layers)
        self.num_of_conv_layers = base_model.num_of_conv_layers
        self.skipping = base_model.skipping

def count_parameters(model):
    """Count number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compress_model(base_model, compression_ratio):
    """
    Compress the model using specified compression ratio
    compression_ratio: float between 0 and 1
    """
    # Create compressed model
    compressed_model = CompressedHCNN(base_model, compression_ratio)
    return compressed_model

# Example usage:
def test_compression(model, compression_ratio=0.5):
    """
    Test the compression of a model
    Args:
        model: original HCNN model
        compression_ratio: float between 0 and 1 (e.g., 0.5 means keep 50% of parameters)
    """
    # Create compressed model
    compressed_model = compress_model(model, compression_ratio)
    
    # Compare parameters
    orig_params = count_parameters(model)
    comp_params = count_parameters(compressed_model)
    
    print(f"Original parameters: {orig_params:,}")
    print(f"Compressed parameters: {comp_params:,}")
    print(f"Compression ratio: {comp_params/orig_params:.2%}")
    
    return compressed_model

