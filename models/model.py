import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CosineHead(nn.Module):
    """L2-normalized cosine head with learnable temperature scalar s."""
    def __init__(self, in_dim: int, out_dim: int, s_init: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self._s_unconstrained = nn.Parameter(torch.tensor(s_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        s = F.softplus(self._s_unconstrained) + 1e-6  # ensure s > 0
        return s * (x @ w.t())


class ScaleGate(nn.Module):
    """Learned per-sample gating over S views."""
    def __init__(self, d_embed: int, n_scales: int = 2, hidden: int | None = None):
        super().__init__()
        h = hidden or max(128, d_embed // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(n_scales * d_embed),
            nn.Linear(n_scales * d_embed, h),
            nn.GELU(),
            nn.Linear(h, n_scales),
        )
        # Zero init → starts at uniform [0.5, 0.5] (same as avg)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, P: torch.Tensor):
        B, S, D = P.shape
        g = F.softmax(self.net(P.reshape(B, S * D)), dim=-1)
        fused = torch.sum(P * g.unsqueeze(-1), dim=1)
        return fused, g


class MultiViewClassifier(nn.Module):
    """
    Multi-view classifier with four fusion modes:
      avg:    mean(emb_a, emb_b)              → Linear(d, C)
      concat: cat(emb_a, emb_b)               → Linear(2d, C)
      gate:   gate_weight * emb_a + w*emb_b   → Linear(d, C)
      late:   avg(head_a(emb_a), head_b(emb_b))  [separate heads]
    """

    FUSE_MODES = ("avg", "concat", "gate", "late")

    def __init__(self, model_name: str, num_classes: int, img_size: int,
                 pretrained: bool, fuse: str, use_cosine_head: bool = False, 
                 cons_T: float = 3.0, verbose: bool = False):
        super().__init__()
        self.fuse = fuse
        self.cons_T = cons_T
        assert fuse in self.FUSE_MODES, f"Unknown fuse={fuse}"

        if verbose:
            print(f"Configuring {model_name} with img_size={img_size} and use_cosine_head={use_cosine_head}", flush=True)

        create_kwargs = dict(pretrained=pretrained, num_classes=0)
        try:
            self.backbone = timm.create_model(model_name, **create_kwargs, img_size=img_size)
        except TypeError:
            self.backbone = timm.create_model(model_name, **create_kwargs)

        self.d = getattr(self.backbone, "num_features", None)

        if self.d is None:
            with torch.no_grad():
                x = torch.randn(2, 3, img_size, img_size)
                emb = self._forward_backbone(x)
                self.d = int(emb.shape[-1])

        def build_head(in_dim, out_dim):
            if use_cosine_head:
                return CosineHead(in_dim, out_dim)
            return nn.Linear(in_dim, out_dim)

        self.gate = None
        self.head_b = None
        if fuse == "concat":
            self.head = build_head(self.d * 2, num_classes)
        elif fuse == "gate":
            self.gate = ScaleGate(self.d, n_scales=2)
            self.head = build_head(self.d, num_classes)
        elif fuse == "late":
            self.gate = ScaleGate(self.d, n_scales=2)
            self.head = build_head(self.d, num_classes)    # view_a head
            self.head_b = build_head(self.d, num_classes)  # view_b head
        else:  # avg
            self.head = build_head(self.d, num_classes)

        # Per-view LayerNorm (2.5x and 10x have different feature distributions)
        self.emb_norm_a = nn.LayerNorm(self.d)
        self.emb_norm_b = nn.LayerNorm(self.d)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        if feats.ndim == 3:
            if getattr(self.backbone, "num_prefix_tokens", 0) > 0:
                return feats[:, 0]
            return feats.mean(dim=1)
        if feats.ndim == 4:
            return feats.mean(dim=(2, 3))
        return feats

    def forward(self, x: torch.Tensor):
        B, S, C, H, W = x.shape
        xs = x.reshape(B * S, C, H, W)
        emb_all = self._forward_backbone(xs)
        P = emb_all.reshape(B, S, -1)

        # Per-view normalization (2.5x and 10x have different distributions)
        P = torch.stack([self.emb_norm_a(P[:, 0]),
                         self.emb_norm_b(P[:, 1])], dim=1)

        aux = {}
        # We no longer strictly need "emb_views" for Cosine loss, but we keep it for downstream 
        # metric logging (cv_cos) if needed.
        aux["emb_views"] = P                        # [B, 2, d] 

        # Calculate per-view logits for KL consistency loss.
        if self.fuse == "late":
            z0 = self.head(P[:, 0])
            z1 = self.head_b(P[:, 1])
        else:
            # For avg, concat, gate, we use the shared head to evaluate each view independently 
            # for the auxiliary consistency loss.
            # However, for 'concat', the head expects [B, 2*d]. We cannot easily get single-view
            # logits through a concat head. We can either return None for z_list, or pad with zeros.
            # Best practice for concat is to skip KL loss, as the head semantics differ.
            if self.fuse == "concat":
                z0 = None
                z1 = None
            else:
                z0 = self.head(P[:, 0])
                z1 = self.head(P[:, 1])
                
        if z0 is not None and z1 is not None:
            aux["z_list"] = [z0, z1]

        if self.fuse == "avg":
            emb = P.mean(dim=1)
            logits = self.head(emb)
        elif self.fuse == "concat":
            emb = P.reshape(B, -1)
            logits = self.head(emb)
        elif self.fuse == "gate":
            emb, g = self.gate(P)
            logits = self.head(emb)
            aux["gate_weights"] = g
        elif self.fuse == "late":
            _, g = self.gate(P)                              # [B, 2]
            logits_a = z0                                    # 2.5x head
            logits_b = z1                                    # 10x head
            # Numerically stable log-space mixture via logsumexp
            logp_a = F.log_softmax(logits_a, dim=-1)         # [B, C]
            logp_b = F.log_softmax(logits_b, dim=-1)         # [B, C]
            eps = torch.finfo(g.dtype).tiny
            logg = torch.log(g.clamp_min(eps))               # [B, 2]
            stack = torch.stack([logg[:, 0:1] + logp_a,
                                 logg[:, 1:2] + logp_b], dim=0)  # [2, B, C]
            logits = torch.logsumexp(stack, dim=0)            # [B, C]
            # Gate-weighted embedding (consistent with classification)
            emb = g[:, 0:1] * P[:, 0] + g[:, 1:2] * P[:, 1]
            aux["gate_weights"] = g
            aux["is_log_prob"] = True  # signal to use nll_loss

        return logits, emb, aux
