import torch
import torch.nn as nn

def extract_color_patches(img, patch_size=4, stride=4):
    """
    Estrae patch 4x4 da un'immagine B,C,H,W con stride specificato e le restituisce
    nel formato B, N_patches, C, 16 (dove 16 sono i valori dei pixel per patch per canale).

    Args:
        img (torch.Tensor): Tensor di input con shape (B, C, H, W)
        patch_size (int): Dimensione della patch (default 4)
        stride (int): Stride per l'estrazione delle patch

    Returns:
        torch.Tensor: Tensor di output con shape (B, N_patches, C, 16)
    """
    B, C, _, _ = img.shape

    # Estrai patch su H e W separatamente per ottenere (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
    patches = img.unfold(2, patch_size, stride).unfold(3, patch_size, stride)  # (B, C, H_out, W_out, patch_size, patch_size)

    # Combina le dimensioni spatiali delle patch
    patches = patches.contiguous().view(B, C, -1, patch_size * patch_size)  # (B, C, N_patches, 16)

    # Riordina in (B, N_patches, C, 16)
    patches = patches.permute(0, 2, 1, 3)  # (B, N_patches, C, 16)

    return patches


def extract_multivector_by_channel(img, patch_size=4, stride=4):
    B, C, _, _ = img.shape

    # Estrai patch su H e W separatamente per ottenere (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
    patches = img.unfold(2, patch_size, stride).unfold(3, patch_size, stride)  # (B, C, H_out, W_out, patch_size, patch_size)

    B, C, H_out, W_out, patch_size, patch_size = patches.shape

    patches = patches.contiguous().view(B, C, H_out * W_out, patch_size * patch_size)
    patches = patches.permute(0, 2, 3, 1)  # (B, 16, N_patches, C)

    return patches


def extract_3d_windows(x, size, window_size=(2, 2, 2), stride=(2, 2, 2)):
    # x: (B, C, H, W)
    B, C, _, _ = x.shape
    x = x.unsqueeze(1)  # (B, 1, C, H, W)
    B, C, _, _, _ = x.shape
    d_win, h_win, w_win = window_size

    patches = x.unfold(2, d_win, stride[0]) \
               .unfold(3, h_win, stride[1]) \
               .unfold(4, w_win, stride[2])  # (B, C, num_d, num_h, num_w, d_win, h_win, w_win)

    num_d, num_h, num_w = patches.shape[2:5]
    patch_size = d_win * h_win * w_win
    num_patches = num_d * num_h * num_w

    patches = patches.contiguous().view(B, C, num_patches, patch_size)
    return patches


def extract_3D_multivector(img, window_size, crop_size, stride=(2,2,2), device="cpu"):
    B, _, _, _ = img.shape
    x = img.unsqueeze(1)  # diventa (B, 1, C, H, W)
    # Definisco il pooling 3D con kernel 2x2x2 e stride 2
    max_pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=stride)

    out_max = max_pool(x)  # (B, 1, C/2, H/2, W/2)
    out_min = -max_pool(-x)  # (B, 1, C/2, H/2, W/2)

    out_max = out_max.squeeze(1) # (B, C/2, H/2, W/2)
    out_min = out_min.squeeze(1) # (B, C/2, H/2, W/2)

    patches_max = extract_3d_windows(out_max, window_size)  # (B, n, 8)
    patches_min = extract_3d_windows(out_min, window_size)  # (B, n, 8)

    # Crea array finale alternato
    patches = torch.stack([patches_max, patches_min], dim=-1)  # (B, 16, 16, 8, 2)
    patches = patches.view(B, crop_size * crop_size, 16, -1)  # (B, 16, 16, 16)
    patches = patches.to(device)

    patches = patches.permute(0, 1, 3, 2)
    return patches


class HGATrEmbeddingTriplePatchToMultivector(nn.Module):
    def __init__(self, window_size, crop_size, device="cpu"):
        super().__init__()

        self.window_size = window_size
        self.crop_size = crop_size
        self.device = device

    def forward(self, x):

        outputs = []

        x_1 = x[0]
        x_2 = x[1]
        x_3 = x[2]

        if any(w is not None for w in x_1):
            x_1 = torch.stack(x_1, dim=0)
            res_1 = extract_color_patches(x_1, 4, 4)
            outputs.append(res_1)

        if any(w is not None for w in x_2):
            x_2 = torch.stack(x_2, dim=0)
            res_2 = extract_multivector_by_channel(x_2, 4, 4)
            _, _, _, C = res_2.shape
            chunks = torch.chunk(res_2, C // 16, dim=-1)  # ogni pezzo ha 16 canali
            outputs.extend(chunks)

        if any(w is not None for w in x_3):
            x_3 = torch.stack(x_3, dim=0)
            res_3 = extract_3D_multivector(x_3, self.window_size, self.crop_size, device=self.device)
            outputs.append(res_3)

        res = torch.cat(outputs, dim=-2)

        return res
