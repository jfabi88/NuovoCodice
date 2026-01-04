import torch
import torch.nn as nn
import torch.nn.functional as F
from hgatr.model.preprocessing import HGATrEmbeddingTriplePatchToMultivector
from gatr import SelfAttentionConfig, GATr, MLPConfig
from hgatr.ga.equilinear import EquiLinearLayer

class HGatr(nn.Module):
    def __init__(
            self,
            in_channels,
            mv_in_channels,
            out_channels,
            blade,
            hidden_dim,
            crop_size,
            n_heads,
            positional_dim = 0,
            blocks = 1,
            dropout_gatr = 0,
            dropout_final = 0.0,
            window_size = 16,
            device = "cpu",
        ):

        super(HGatr, self).__init__()

        self.positional_dim = positional_dim
        self.crop_size = crop_size

        self.preprocessor = HGATrEmbeddingTriplePatchToMultivector(window_size, crop_size)

        self.mv_channels = mv_in_channels

        att_config = SelfAttentionConfig()
        att_config.num_heads = n_heads
        att_config.pos_encoding = False


        if positional_dim != 0:
          self.gatr = GATr(
            in_mv_channels = in_channels,
            out_mv_channels = out_channels,
            hidden_mv_channels = hidden_dim,
            in_s_channels = positional_dim,
            out_s_channels = 1,
            hidden_s_channels = 4,
            attention = att_config,
            mlp = MLPConfig(),
            num_blocks = blocks,
            dropout_prob = dropout_gatr,
          )
        else:
          self.gatr = GATr(
            in_mv_channels = in_channels,
            out_mv_channels = out_channels,
            hidden_mv_channels = hidden_dim,
            attention = att_config,
            mlp = MLPConfig(),
            num_blocks = blocks,
            dropout_prob = dropout_gatr,
          )

        if positional_dim:
          self.position_encoding = nn.Parameter(torch.randn(1, (crop_size * crop_size), positional_dim)) # <- levato il +1 15/07/2025

        self._initialize_weights()
    
        self.classification = EquiLinearLayer(blade, 9, (crop_size * crop_size), 1, dropout_p = dropout_final, device = device)

        self.vectorizer = nn.Linear(
             16,
             1,
        )


    def _initialize_weights(self):
        """
        Funzione per inizializzare i pesi dei moduli del modello.
        Può essere estesa con altre strategie di inizializzazione.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Inizializzazione di Xavier (adatta per attivazioni ReLU)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # Inizializzazione di He per i layer convoluzionali
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # Inizializzazione della normalizzazione del layer
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x, scalars = self.preprocessor(x)

        if self.positional_dim:
          positional_emb = self.position_encoding.reshape(x.shape[1], self.positional_dim)
          positional_emb = positional_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
          if scalars != None:
            positional_emb = torch.cat((positional_emb, scalars), dim=-1)

          out_mv, _ = self.gatr(x, scalars = positional_emb)
        else:
          out_mv, _ = self.gatr(x, scalars)
        
        out = out_mv.permute(0, 2, 1, -1)
        out = self.classification(out)
        out = out.squeeze(-2)

        out = F.dropout(out, p = 0.3, training = self.training)
        out = self.vectorizer(out)
        y = out.squeeze(-1)

        return y