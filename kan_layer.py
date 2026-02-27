import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLayer(nn.Module):
    """
    A minimal 1D Kolmogorov-Arnold Network (KAN) Layer in pure PyTorch.
    This layer replaces the standard W*x + b linear transformation with a sum of 
    learnable 1D functions (parameterized as B-splines) for each input-output pair.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Grid definition [grid_size + 2 * spline_order + 1]
        step = 2 / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * step - 1
        self.register_buffer('grid', grid)
        
        # Base weights (similar to standard linear layer but applies to SiLU(x))
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Spline coefficients [out_features, in_features, grid_size + spline_order]
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        
        self.reset_parameters(scale_noise)

    def reset_parameters(self, scale_noise):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5)) if hasattr(math, 'sqrt') else nn.init.uniform_(self.base_weight, -0.1, 0.1) 
        # Note: Added simple uniform fallback if math is not imported, though usually it is. We'll add import math at top.
        nn.init.normal_(self.spline_weight, mean=0.0, std=scale_noise)

    def b_spline(self, x):
        """
        Evaluate the B-spline basis functions for input x.
        x: [batch, in_features]
        Returns: [batch, in_features, grid_size + spline_order]
        """
        # Add dimensions for broadcasting: x shape becomes [batch, in_features, 1]
        x = x.unsqueeze(-1)
        
        # Basis initialization (degree 0)
        bases = ((x >= self.grid[:-1]) & (x < self.grid[1:])).to(x.dtype)
        
        # Cox-de Boor recursion for higher degrees
        for k in range(1, self.spline_order + 1):
            # Left term
            left_num = x - self.grid[:-k-1]
            left_den = self.grid[k:-1] - self.grid[:-k-1]
            # Avoid division by zero
            left = (left_num / left_den) * bases[:, :, :-1]
            
            # Right term
            right_num = self.grid[k+1:] - x
            right_den = self.grid[k+1:] - self.grid[1:-k]
            # Avoid division by zero
            right = (right_num / right_den) * bases[:, :, 1:]
            
            bases = left + right
            
        return bases

    def forward(self, x):
        """
        Forward pass of the KAN layer.
        x: [batch, in_features]
        Output: [batch, out_features]
        """
        batch_size = x.shape[0]
        
        # 1. Base activation (Standard SiLU mapping)
        base_output = F.linear(F.silu(x), self.base_weight)
        
        # 2. Spline activation
        # Evaluate B-spline bases for all inputs: [batch, in_features, num_splines]
        splines = self.b_spline(x)
        
        # Reshape to easily multiply with weights: [batch, in_features * num_splines]
        splines = splines.view(batch_size, -1)
        
        # Reshape spline weights: [out_features, in_features * num_splines]
        spline_weight_flat = self.spline_weight.view(self.out_features, -1)
        
        # Linear combination of splines
        spline_output = F.linear(splines, spline_weight_flat)
        
        # 3. Combine base and spline outptus
        return self.scale_base * base_output + self.scale_spline * spline_output

if __name__ == "__main__":
    # Test the minimal implementation
    batch_size = 32
    in_features = 4
    out_features = 2
    
    # Create the layer
    kan_layer = KANLayer(in_features, out_features, grid_size=5, spline_order=3)
    
    # Dummy input
    x = torch.randn(batch_size, in_features)
    
    # Forward pass
    y = kan_layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Success! Output matches expected shape: {list((batch_size, out_features)) == list(y.shape)}")
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in kan_layer.parameters() if p.requires_grad)
    print(f"Total Parameters in layer: {total_params}")
