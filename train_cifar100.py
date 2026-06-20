# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torchvision",
#     "accelerate",
#     "x-transformers",
#     "einops",
#     "fire"
# ]
# ///

import fire

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from einops import rearrange, repeat

from accelerate import Accelerator
from x_transformers import FeedForward

from rotary_embedding_torch import RotaryEmbedding
from rotary_embedding_torch.flash_attn_with_rotary import flash_attn_with_rotary

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

# classes

class FusedRotaryAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        rotary_pos_emb = None,
        rotary_pos_emb_indices = None
    ):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        out = flash_attn_with_rotary(
            q, k, v,
            rotary_pos_emb = rotary_pos_emb,
            rotary_pos_emb_indices = rotary_pos_emb_indices,
            is_causal = False
        )

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                FusedRotaryAttention(dim, heads = heads, dim_head = dim_head),
                nn.LayerNorm(dim),
                FeedForward(dim, dim_out = dim, mult = mlp_dim / dim)
            ]))

    def forward(
        self,
        x,
        rotary_pos_emb = None,
        rotary_pos_emb_indices = None
    ):
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x), rotary_pos_emb = rotary_pos_emb, rotary_pos_emb_indices = rotary_pos_emb_indices) + x
            x = ff(norm2(x)) + x
        return x

class SimpleViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim
    ):
        super().__init__()
        assert divisible_by(image_size, patch_size), 'image dimensions must be divisible by the patch size'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.grid_size = image_size // patch_size

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(dim))
        self.rotary = RotaryEmbedding(dim // 2)
        self.transformer = Transformer(dim, depth, heads, dim // heads, mlp_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def get_2d_rotary_embeddings(self, device):
        freqs_2d = self.rotary.get_axial_freqs(self.grid_size, self.grid_size)
        indices = torch.arange(1, self.num_patches + 1, device = device)
        return freqs_2d, indices

    def forward(self, img):
        b, p = img.shape[0], self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = p, p2 = p)
        x = self.to_patch_embedding(x)

        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        rotary_pos_emb, rotary_pos_emb_indices = self.get_2d_rotary_embeddings(img.device)

        x = self.transformer(
            x,
            rotary_pos_emb = rotary_pos_emb,
            rotary_pos_emb_indices = rotary_pos_emb_indices
        )

        return self.mlp_head(x[:, 0])

def train(
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 0.01
):
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Using device: {device}")
    accelerator.print("Note: CLS token is omitted from relative positions, while the rest of the tokens get 2D axial rotary embeddings.")

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = SimpleViT(
        image_size=32,
        patch_size=4,
        num_classes=100,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    accelerator.print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 10 == 0:
                accelerator.print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")

        accelerator.print(f"--- Epoch {epoch+1} Summary --- | Avg Loss: {total_loss/(batch_idx + 1):.4f} | Train Acc: {100.*correct/total:.2f}%")

if __name__ == '__main__':
    fire.Fire(train)
