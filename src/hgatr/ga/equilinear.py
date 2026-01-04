import torch
import torch.nn as nn

class EquiLinearLayer(nn.Module):
    """
    Equivariant Linear Layer.

    Args:
        input_mv_channels (int): number of input channels in the multivector
        hidden_mv_dim (int): number of output channels in the multivector
        blade (torch.Tensor): blade tensor representing the geometric entity
        blade_len (int): length of the blade tensor

    Attributes:
        blade (torch.Tensor): blade tensor representing blade operator
        weights (nn.Parameter): learnable weights for the linear layer

    Methods:
        forward(x): computes the forward pass of the equivariant linear layer
    """

    def __init__(self, blade, blade_len, in_mv_channels, out_mv_channels, dropout_p = 0.0, device="cpu"):
        super(EquiLinearLayer,self).__init__()

        self.blade = blade

        self.weights = nn.Parameter(
            torch.rand(out_mv_channels, in_mv_channels, blade_len, device=device)
         )

        self.dropout = nn.Dropout(p=dropout_p).to(device)


    def forward(self, multivectors):
        """
        Parameters
        ----------
            multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors
            scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """

        multivectors = self.dropout(multivectors)
        outputs_mv = torch.einsum(
            "j i b, b x y, ... i x -> ... j y",
            self.weights,
            self.blade,
            multivectors
        )

        return outputs_mv