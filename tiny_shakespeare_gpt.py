"""
Tiny Shakespeare GPT with TensorBoard dashboard.
Character-level language model based on Karpathy's nanoGPT.
"""

import os
import math
import time

import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

# ── Hyperparameters ───────────────────────────────────────────────────────────
BLOCK_SIZE   = 128
BATCH_SIZE   = 32
N_EMBD       = 64
N_HEAD       = 4
N_LAYER      = 4
DROPOUT      = 0.1
LR           = 3e-4
MAX_ITERS    = 5000
GRAD_CLIP    = 1.0
EVAL_EVERY   = 200
EVAL_BATCHES = 20
GEN_EVERY    = 1000
GEN_LEN      = 300
FINAL_LEN    = 500
MODEL_PATH   = "tiny_shakespeare.pt"
DATA_URL     = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")

# ── Download dataset ──────────────────────────────────────────────────────────
DATA_PATH = "tinyshakespeare.txt"
if not os.path.exists(DATA_PATH):
    print("Downloading dataset...")
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    with open(DATA_PATH, "w") as f:
        f.write(r.text)
    print(f"Saved {len(r.text):,} chars to {DATA_PATH}\n")
else:
    print(f"Dataset already exists: {DATA_PATH}\n")

with open(DATA_PATH, "r") as f:
    text = f.read()

# ── Character-level tokenization ──────────────────────────────────────────────
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

print(f"Vocab size: {vocab_size} characters")
print(f"Dataset:    {len(text):,} characters\n")

# ── Train / val split ─────────────────────────────────────────────────────────
data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]

def get_batch(split_name):
    d = train_data if split_name == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i+BLOCK_SIZE]   for i in ix]).to(DEVICE)
    y = torch.stack([d[i+1:i+BLOCK_SIZE+1] for i in ix]).to(DEVICE)
    return x, y

# ── Model ─────────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        self.c_attn  = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj  = nn.Linear(N_EMBD, N_EMBD)
        self.attn_drop = nn.Dropout(DROPOUT)
        self.resid_drop = nn.Dropout(DROPOUT)
        self.n_head  = N_HEAD
        self.head_dim = N_EMBD // N_HEAD
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
               .view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(N_EMBD, dim=2)
        # reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(N_EMBD)
        self.mlp  = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.head    = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=DEVICE)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ── Loss estimation ───────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split_name in ("train", "val"):
        ls = torch.zeros(EVAL_BATCHES)
        for k in range(EVAL_BATCHES):
            x, y = get_batch(split_name)
            _, loss = model(x, y)
            ls[k] = loss.item()
        losses[split_name] = ls.mean().item()
    model.train()
    return losses


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


# ── Training ──────────────────────────────────────────────────────────────────
model = GPT().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
writer = SummaryWriter(log_dir="runs/tiny_shakespeare")

print(f"{'Step':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Grad Norm':>10}  {'LR':>10}")
print("-" * 55)

t0 = time.time()

for step in range(1, MAX_ITERS + 1):
    # Forward + backward
    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient clipping → record norm BEFORE clip
    gn = grad_norm(model)
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    optimizer.step()
    current_lr = optimizer.param_groups[0]["lr"]

    # ── Eval + logging every EVAL_EVERY steps ─────────────────────────────
    if step % EVAL_EVERY == 0:
        losses = estimate_loss(model)

        # TensorBoard
        writer.add_scalar("loss/train",    losses["train"], step)
        writer.add_scalar("loss/val",      losses["val"],   step)
        writer.add_scalar("grad_norm",     gn,              step)
        writer.add_scalar("learning_rate", current_lr,      step)

        elapsed = time.time() - t0
        print(
            f"{step:>6}  {losses['train']:>10.4f}  {losses['val']:>10.4f}"
            f"  {gn:>10.4f}  {current_lr:>10.2e}  ({elapsed:.1f}s)"
        )

    # ── Generate sample every GEN_EVERY steps ─────────────────────────────
    if step % GEN_EVERY == 0:
        model.eval()
        ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        sample = decode(model.generate(ctx, GEN_LEN)[0].tolist())
        model.train()
        print(f"\n── Sample at step {step} ──────────────────────────────────")
        print(sample)
        print("─" * 55 + "\n")

writer.close()
print("\nTraining complete.\n")

# ── Final sample ──────────────────────────────────────────────────────────────
model.eval()
ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
final_sample = decode(model.generate(ctx, FINAL_LEN)[0].tolist())
print("═" * 55)
print("FINAL 500-CHAR SAMPLE")
print("═" * 55)
print(final_sample)
print("═" * 55 + "\n")

# ── Save model ────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved → {MODEL_PATH}")

# ── TensorBoard reminder ──────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("  TensorBoard dashboard:")
print("    tensorboard --logdir=runs")
print("    → http://localhost:6006")
print("─" * 55)
