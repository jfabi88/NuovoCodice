from dataclasses import dataclass

@dataclass
class Parameters:
    h_ch: int = 16
    ver_ch: int = 16
    vol_ch: int = 16
    split: float = 0.7
    batch_size: int = 8
    in_channels: int = 1
    out_channels: int = 13
    hidden_dim: int = 8
    n_heads: int = 8
    n_classes: 13
    positional_dim: 128
    mv_in_channels: 48
    blocks: int = 1
    dropout_gatr: float = 0.4
    dropout_final: float = 0.2
    lr: float = 0.001
    max_epochs: int = 400
    patience: int = 20
    window_size: int = 16
