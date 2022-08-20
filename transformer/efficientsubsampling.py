# PyTorch
import torch
import torch.nn as nn
import torchaudio


###############################################################################
# Conv Subsampling Modules
###############################################################################

class Conv1dSubsampling(nn.Module):

    """Conv1d Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        in_dim: input feature dimension
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (batch_size, in_dim, in_length)
        Output: (batch_size, out_dim, out_length)
    
    """

    def __init__(self, num_layers, in_dim, filters, kernel_size, norm, act):
        super(Conv1dSubsampling, self).__init__()

        # Assert
        assert norm in ["batch", "layer", "none"]
        assert act in ["relu", "swish", "none"]

        # Layers
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_dim if layer_id == 0 else filters[layer_id - 1], filters[layer_id], kernel_size, stride=2, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(filters[layer_id]) if norm == "batch" else nn.LayerNorm(filters[layer_id]) if norm == "layer" else nn.Identity(),
            nn.ReLU() if act == "relu" else Swish() if act == "swish" else nn.Identity()
        ) for layer_id in range(num_layers)])

    def forward(self, x, x_len):

        # Layers
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = torch.div(x_len - 1, 2, rounding_mode='floor') + 1

        return x, x_len

class Conv2dSubsampling(nn.Module):

    """Conv2d Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (batch_size, in_dim, in_length)
        Output: (batch_size, out_dim, out_length)
    
    """

    def __init__(self, num_layers, filters, kernel_size, norm, act):
        super(Conv2dSubsampling, self).__init__()

        # Assert
        assert norm in ["batch", "layer", "none"]
        assert act in ["relu", "swish", "none"]

        # Conv 2D Subsampling Layers
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1 if layer_id == 0 else filters[layer_id - 1], filters[layer_id], kernel_size, stride=2, padding=(kernel_size - 1) // 2), 
            nn.BatchNorm2d(filters[layer_id]) if norm == "batch" else nn.LayerNorm(filters[layer_id]) if norm == "layer" else nn.Identity(),
            nn.ReLU() if act == "relu" else Swish() if act == "swish" else nn.Identity()
        ) for layer_id in range(num_layers)])

    def forward(self, x, x_len, x_mask):

        # (B, D, T) -> (B, 1, D, T)
        x = x.unsqueeze(dim=1)

        # Layers
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = torch.div(x_len - 1, 2, rounding_mode='floor') + 1

        # (B, C, D // S, T // S) -> (B,  C * D // S, T // S)
        batch_size, channels, subsampled_dim, subsampled_length = x.size()
        x = x.reshape(batch_size, channels * subsampled_dim, subsampled_length)

        return x, x_len, x_mask[:, :, :-2:2][:, :, :-2:2]

class Conv2dPoolSubsampling(nn.Module):

    """Conv2d with Max Pooling Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (batch_size, in_dim, in_length)
        Output: (batch_size, out_dim, out_length)
    
    """

    def __init__(self, num_layers, filters, kernel_size, norm, act):
        super(Conv2dPoolSubsampling, self).__init__()

        # Assert
        assert norm in ["batch", "layer", "none"]
        assert act in ["relu", "swish", "none"]

        # Layers
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1 if layer_id == 0 else filters[layer_id - 1], filters[layer_id], kernel_size, padding=(kernel_size - 1) // 2),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(filters[layer_id]) if norm == "batch" else nn.LayerNorm(filters[layer_id]) if norm == "layer" else nn.Identity(),
            nn.ReLU() if act == "relu" else Swish() if act == "swish" else nn.Identity()
        ) for layer_id in range(num_layers)])

    def forward(self, x, x_len):

        # (B, D, T) -> (B, 1, D, T)
        x = x.unsqueeze(dim=1)

        # Layers
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = torch.div(x_len - 1, 2, rounding_mode='floor') + 1

        # (B, C, D // S, T // S) -> (B,  C * D // S, T // S)
        batch_size, channels, subsampled_dim, subsampled_length = x.size()
        x = x.reshape(batch_size, channels * subsampled_dim, subsampled_length)

        return x, x_len

class VGGSubsampling(nn.Module):

    """VGG style Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (batch_size, in_dim, in_length)
        Output: (batch_size, out_dim, out_length)
    
    """

    def __init__(self, num_layers, filters, kernel_size, norm, act):
        super(VGGSubsampling, self).__init__()

        # Assert
        assert norm in ["batch", "layer", "none"]
        assert act in ["relu", "swish", "none"]

        self.layers = nn.ModuleList([nn.Sequential(
            # Conv 1
            nn.Conv2d(1 if layer_id == 0 else filters[layer_id - 1], filters[layer_id], kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(filters[layer_id]) if norm == "batch" else nn.LayerNorm(filters[layer_id]) if norm == "layer" else nn.Identity(),
            nn.ReLU() if act == "relu" else Swish() if act == "swish" else nn.Identity(),
            # Conv 2
            nn.Conv2d(filters[layer_id], filters[layer_id], kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(filters[layer_id]) if norm == "batch" else nn.LayerNorm(filters[layer_id]) if norm == "layer" else nn.Identity(),
            nn.ReLU() if act == "relu" else Swish() if act == "swish" else nn.Identity(),
            # Pooling
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) 
        ) for layer_id in range(num_layers)])

    def forward(self, x, x_len):

        # (B, D, T) -> (B, 1, D, T)
        x = x.unsqueeze(dim=1)

        # Stages
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = torch.div(x_len, 2, rounding_mode='floor')

        # (B, C, D // S, T // S) -> (B,  C * D // S, T // S)
        batch_size, channels, subsampled_dim, subsampled_length = x.size()
        x = x.reshape(batch_size, channels * subsampled_dim, subsampled_length)
        
        return x, x_len


class StreamingMask(nn.Module):

    def __init__(self, left_context, right_context):
        super(StreamingMask, self).__init__()
        self.padding_mask = PaddingMask()
        self.left_context = left_context
        self.right_context = right_context

    def forward(self, x, x_len):

        # Seq Length T
        seq_len = x.size(-1)

        # Right Context Mask (T, T)
        right_context_mask = x.new_ones(seq_len, seq_len).triu(diagonal=1+self.right_context)

        # Left Context Mask (T, T)
        left_context_mask = 1 - x.new_ones(seq_len, seq_len).triu(diagonal=-self.left_context)

        # Streaming Mask (T, T)
        streaming_mask = right_context_mask.max(left_context_mask)

        # Padding Mask
        if x_len is not None:

            # Padding Mask (B, 1, 1, T)
            padding_mask = self.padding_mask(seq_len, x_len)

            # Streaming Mask + Padding Mask (B, 1, T, T)
            return streaming_mask.maximum(padding_mask)
        
        else:

            # Streaming Mask (1, 1, T, T)
            return streaming_mask[None, None, :, :]
