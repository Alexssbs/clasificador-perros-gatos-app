"""Arquitectura extraída del código original de entrenamiento, sin ejecutar el notebook."""
import torch
import torch.nn as nn


def load_vit_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("El archivo ViT debe contener model_state_dict y config.")
    config = checkpoint.get("config", {})
    if config.get("clases", ["cat", "dog"]) != ["cat", "dog"]:
        raise ValueError("El ViT no tiene el orden de clases esperado: cat, dog.")
    # El export incluye metadatos que no son argumentos del constructor.
    defaults = dict(image_size=224, patch_size=16, in_channels=3,
                    num_classes=2, embed_dim=192, depth=6,
                    num_heads=3, mlp_dim=768, dropout=0.1)
    parameters = {key: config.get(key, value) for key, value in defaults.items()}
    if parameters["image_size"] != 224 or parameters["num_classes"] != 2:
        raise ValueError("La aplicación requiere entrada 224x224 y dos clases.")
    model = VisionTransformer(**parameters)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim debe ser divisible entre num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_size, num_tokens, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale
        attention = scores.softmax(dim=-1)
        attention = self.attn_dropout(attention)

        out = attention @ v
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out

class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(
            embed_dim, num_heads, dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        # Conexión residual alrededor de la atención
        x = x + self.attention(self.norm1(x))
        # Conexión residual alrededor del MLP
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_dim=768,
        dropout=0.1
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size debe ser divisible por patch_size.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_side = image_size // patch_size
        self.num_patches = self.num_patches_side ** 2

        # La convolución con kernel y stride iguales al tamaño del parche
        # convierte cada parche en un vector de dimensión embed_dim.
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size = x.shape[0]

        # [B, 3, 224, 224] -> [B, embed_dim, 14, 14]
        x = self.patch_embed(x)

        # [B, embed_dim, 14, 14] -> [B, 196, embed_dim]
        x = x.flatten(2).transpose(1, 2)

        # Añadir token CLS al principio de la secuencia
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Añadir información posicional
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Encoder Transformer
        x = self.blocks(x)
        x = self.norm(x)

        # Usar la representación del token CLS para clasificar
        cls_representation = x[:, 0]
        logits = self.head(cls_representation)
        return logits
