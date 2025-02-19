import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

# from models import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0)#.transpose(0, 1)

        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[1], :].unsqueeze(0)
        return self.dropout(x)

class MotionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.input_feats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear(self.input_feats*2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=2000)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True)
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)


    def forward(self, batch):
        x, mask = batch["motions"], batch["mask"]
        B, T, D  = x.shape

        x = x.reshape(B, T, 2, -1)[..., :-4].reshape(B, T, -1)

        x_emb = self.embed_motion(x)

        emb = torch.cat([self.query_token[torch.zeros(B, dtype=torch.long, device=x.device)][:,None], x_emb], dim=1)

        seq_mask = (mask>0.5)
        token_mask = torch.ones((B, 1), dtype=bool, device=x.device)
        valid_mask = torch.cat([token_mask, seq_mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        motion_emb = self.out(h[:,0])

        batch["motion_emb"] = motion_emb

        return batch

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

loss_ce = nn.CrossEntropyLoss()
class InterCLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.motion_encoder = MotionEncoder(cfg)

        self.latent_dim = self.latent_dim

        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.dtype = clip_model.dtype
        self.latent_scale = nn.Parameter(torch.Tensor([1]))

        set_requires_grad(self.token_embedding, False)

        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=cfg.FF_SIZE,
            dropout=0.1,
            activation="gelu",
            batch_first=True)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=8)
        self.text_ln = nn.LayerNorm(768)
        self.out = nn.Linear(768, 512)

        self.clip_training = "text_"
        self.l1_criterion = torch.nn.L1Loss(reduction='mean')

    def compute_loss(self, batch):
        losses = {}
        losses["total"] = 0

        # compute clip losses
        batch = self.encode_text(batch)
        batch = self.encode_motion(batch)

        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)
        losses.update(clip_losses)
        losses["total"] += mixed_clip_loss

        return losses["total"], losses

    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_clip_losses(self, batch):
        mixed_clip_loss = 0.
        clip_losses = {}

        if 1:
            for d in self.clip_training.split('_')[:1]:
                if d == 'image':
                    features = self.clip_model.encode_image(batch['images']).float()  # preprocess is done in dataloader
                elif d == 'text':
                    features = batch['text_emb']
                motion_features = batch['motion_emb']
                # normalized features
                features_norm = features / features.norm(dim=-1, keepdim=True)
                motion_features_norm = motion_features / motion_features.norm(dim=-1, keepdim=True)

                logit_scale = self.latent_scale ** 2
                logits_per_motion = logit_scale * motion_features_norm @ features_norm.t()
                logits_per_d = logits_per_motion.t()

                batch_size = motion_features.shape[0]
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=motion_features.device)

                ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
                ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
                clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

                clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss.item()
                clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss.item()
                clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss.item()
                mixed_clip_loss += clip_mixed_loss

        return mixed_clip_loss, clip_losses

    def generate_src_mask(self, T, length):
        B = length.shape[0]
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def encode_motion(self, batch):
        batch["mask"] = self.generate_src_mask(batch["motions"].shape[1], batch["motion_lens"]).to(batch["motions"].device)
        batch.update(self.motion_encoder(batch))
        batch["motion_emb"] = batch["motion_emb"] / batch["motion_emb"].norm(dim=-1, keepdim=True) * self.latent_scale

        return batch

    def encode_text(self, batch):
        device = next(self.parameters()).device
        raw_text = batch["text"]

        with torch.no_grad():
            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self.dtype)

        out = self.textTransEncoder(pe_tokens)
        out = self.text_ln(out)

        out = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        out = self.out(out)

        batch['text_emb'] = out
        batch["text_emb"] = batch["text_emb"] / batch["text_emb"].norm(dim=-1, keepdim=True) * self.latent_scale

        return batch
