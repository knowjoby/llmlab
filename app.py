"""
Shakespeare GPT — Browser App (Gradio)

Tabs:
  🏋️ Train        — train from scratch or load saved weights
  🎭 Generate     — verbose log + confidence colours + entropy chart + branch + verbatim check
  🔬 Score        — paste any text, see per-character confidence
  🌡️ Temperature  — same prompt at 5 temperatures side by side
  ⏪ Replay       — watch the model learn step by step
  🔭 Attention    — see which characters the model focuses on
"""

import os, math, time, json
import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
import gradio as gr
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
BLOCK_SIZE   = 128
BATCH_SIZE   = 32
N_EMBD       = 64
N_HEAD       = 4
N_LAYER      = 4
DROPOUT      = 0.1
GRAD_CLIP    = 1.0
EVAL_BATCHES = 20
MODEL_PATH   = "tiny_shakespeare.pt"
DATA_PATH    = "tinyshakespeare.txt"
DATA_URL     = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR     = "checkpoints"
HISTORY_PATH = os.path.join("runs", "run_history.json")
NGRAM_N      = 6
os.makedirs(CKPT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════
if not os.path.exists(DATA_PATH):
    print("Downloading Shakespeare dataset...")
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    with open(DATA_PATH, "w") as f:
        f.write(r.text)

with open(DATA_PATH) as f:
    _text = f.read()

chars      = sorted(set(_text))
VOCAB_SIZE = len(chars)
stoi       = {c: i for i, c in enumerate(chars)}
itos       = {i: c for i, c in enumerate(chars)}
encode     = lambda s: [stoi[c] for c in s if c in stoi]
decode     = lambda l: "".join(itos[i] for i in l)

_data      = torch.tensor(encode(_text), dtype=torch.long)
_split     = int(0.9 * len(_data))
train_data = _data[:_split]
val_data   = _data[_split:]

def get_batch(split):
    d  = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x  = torch.stack([d[i : i + BLOCK_SIZE]       for i in ix]).to(DEVICE)
    y  = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix]).to(DEVICE)
    return x, y

# Build n-gram index for verbatim detection (runs once at startup)
print("Building vocabulary index...")
_ngrams = set(_text[i:i + NGRAM_N] for i in range(len(_text) - NGRAM_N))
print(f"Ready. {len(_ngrams):,} unique {NGRAM_N}-grams indexed.")

# Fixed test phrase for replay (first 120 chars of training text)
_REPLAY_TEST = _text[:120]

# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn     = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj     = nn.Linear(N_EMBD, N_EMBD)
        self.attn_drop  = nn.Dropout(DROPOUT)
        self.resid_drop = nn.Dropout(DROPOUT)
        self.n_head     = N_HEAD
        self.head_dim   = N_EMBD // N_HEAD
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
               .view(1, 1, BLOCK_SIZE, BLOCK_SIZE),
        )
        self._capture   = False          # set True to record attention weights
        self._last_attn = None           # (n_head, T, T) after forward pass

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(N_EMBD, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        if self._capture:
            self._last_attn = att[0].detach().cpu()   # store for visualisation
        att = self.attn_drop(att)
        y   = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT),
        )
    def forward(self, x): return self.net(x)


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
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.head    = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=DEVICE)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x    = self.blocks(x)
        return self.head(self.ln_f(x))

# ══════════════════════════════════════════════════════════════════════════════
# Global state
# ══════════════════════════════════════════════════════════════════════════════
_model                = None
_optimizer            = None
_gen_history          = []
_last_temp            = 1.0
_last_prompt          = ""
_training_in_progress = False
_run_history          = []
_MAX_HISTORY          = 80
_last_fix_action      = "none"

def _build_model(lr=3e-4):
    global _model, _optimizer
    _model     = GPT().to(DEVICE)
    _optimizer = torch.optim.AdamW(_model.parameters(), lr=float(lr))

def model_status_str():
    if _model is None:
        return "⚠️  No model loaded — use the Train tab."
    n = sum(p.numel() for p in _model.parameters())
    return f"✅  {n:,} params  |  vocab {VOCAB_SIZE}  |  device: {DEVICE}"

# ══════════════════════════════════════════════════════════════════════════════
# Maths helpers
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def estimate_loss(m):
    m.eval()
    out = {}
    for split in ("train", "val"):
        ls = torch.zeros(EVAL_BATCHES)
        for k in range(EVAL_BATCHES):
            x, y   = get_batch(split)
            logits = m(x)
            ls[k]  = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()
        out[split] = ls.mean().item()
    m.train()
    return out

@torch.no_grad()
def compute_perplexity(m, text):
    """Compute perplexity of m on text (up to first 120 chars)."""
    tokens = encode(text)
    if len(tokens) < 2:
        return 999.0
    m.eval()
    log_probs = []
    for i in range(min(len(tokens) - 1, 100)):
        ctx    = tokens[max(0, i + 1 - BLOCK_SIZE): i + 1]
        idx_in = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
        probs  = F.softmax(m(idx_in)[:, -1, :], dim=-1).squeeze(0)
        log_probs.append(math.log(max(probs[tokens[i + 1]].item(), 1e-10)))
    return math.exp(-sum(log_probs) / len(log_probs))

def grad_norm(m):
    total = sum(p.grad.data.norm(2).item() ** 2
                for p in m.parameters() if p.grad is not None)
    return math.sqrt(total)

def entropy_bits(probs):
    p = probs.clamp(min=1e-10)
    return (-(p * p.log2()).sum()).item()

# ══════════════════════════════════════════════════════════════════════════════
# Text helpers
# ══════════════════════════════════════════════════════════════════════════════
def human_char(c):
    names = {'\n': '↵ newline', ' ': '· space', '\t': '→ tab',
             '\r': '↩ return', '.': '. period', ',': ', comma',
             '!': '! exclaim', '?': '? question', ';': '; semicolon',
             ':': ': colon', "'": "' apostrophe", '"': '" quote',
             '-': '- dash', '(': '( open', ')': ') close'}
    return names.get(c, f'"{c}"')

def entropy_label(H):
    if H < 0.5:  return "almost certain"
    if H < 1.5:  return "very confident"
    if H < 2.5:  return "fairly sure"
    if H < 3.2:  return "a bit uncertain"
    if H < 4.0:  return "lots of options"
    return "hard to decide"

def choice_comment(rank, conf):
    if rank == 1 and conf > 0.70: return "its top pick — very confident"
    if rank == 1 and conf > 0.35: return "its top pick"
    if rank == 1:                 return "its top pick (lots of close competition)"
    if rank <= 3:                 return f"a close runner-up (option #{rank})"
    if rank <= 6:                 return f"a less obvious choice (option #{rank})"
    return f"a surprise — the temperature made it adventurous (option #{rank})"

# ══════════════════════════════════════════════════════════════════════════════
# Colour + HTML helpers
# ══════════════════════════════════════════════════════════════════════════════
def conf_to_hex(conf: float) -> str:
    """Confidence 0→1 mapped to hex colour: red → yellow → green."""
    if conf < 0.3:
        t = conf / 0.3
        r, g, b = 220, int(80 + 120 * t), 80
    elif conf < 0.6:
        t = (conf - 0.3) / 0.3
        r, g, b = int(220 - 40 * t), int(200 + 20 * t), int(80 - 44 * t)
    else:
        t = (conf - 0.6) / 0.4
        r, g, b = int(180 - 120 * t), int(220 + 10 * t), int(36 + 90 * t)
    return f"#{r:02x}{g:02x}{b:02x}"

_WRAP = ('<div style="font-family:\'SF Mono\',monospace;font-size:13px;'
         'line-height:2.0;padding:14px;background:#0f1117;color:#e2e8f0;'
         'border-radius:8px;white-space:pre-wrap;word-wrap:break-word;">')
_KEY  = ('<div style="margin-top:10px;font-size:11px;color:#8892b0;">'
         'Colour: <span style="background:#dc504044;padding:1px 7px;border-radius:3px">guessing</span> '
         '<span style="background:#dcc82444;padding:1px 7px;border-radius:3px">uncertain</span> '
         '<span style="background:#3cde7e44;padding:1px 7px;border-radius:3px">confident</span>'
         '</div>')

def _span(c, bg_hex, tip=""):
    tip_attr = f' title="{tip}"' if tip else ""
    if c == '\n': disp = '<span style="opacity:.35">↵</span><br>'
    elif c == ' ': disp = '&nbsp;'
    else: disp = c.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    return f'<span style="background:{bg_hex}44;border-radius:2px;padding:1px 0"{tip_attr}>{disp}</span>'

def make_colored_html(prompt, generated, confidences):
    prompt_safe = (prompt.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
                   .replace('\n','<span style="opacity:.3">↵</span><br>'))
    html = _WRAP + f'<span style="opacity:.45">{prompt_safe}</span>'
    for c, conf in zip(generated, confidences):
        html += _span(c, conf_to_hex(conf), f"{conf*100:.1f}% confident")
    return html + '</div>' + _KEY

def make_verbatim_html(prompt, generated, mask):
    """Render generated text: purple = in training data, teal = genuinely new."""
    prompt_safe = (prompt.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
                   .replace('\n','<span style="opacity:.3">↵</span><br>'))
    html = _WRAP + f'<span style="opacity:.45">{prompt_safe}</span>'
    for c, is_known in zip(generated, mask):
        col = "#a78bfa" if is_known else "#22d3ee"   # purple=familiar, teal=novel
        html += _span(c, col)
    novel_pct = 100 * sum(1 for m in mask if not m) / max(len(mask), 1)
    html += '</div>'
    html += (f'<div style="margin-top:8px;font-size:11px;color:#8892b0;">'
             f'<span style="background:#a78bfa44;padding:1px 7px;border-radius:3px">seen in training data</span> '
             f'<span style="background:#22d3ee44;padding:1px 7px;border-radius:3px">genuinely new</span>'
             f'  —  <strong style="color:#e2e8f0">{novel_pct:.0f}% of this text is original</strong></div>')
    return html

def _dark_ax(ax):
    ax.set_facecolor('#1a1d27')
    ax.tick_params(colors='#8892b0', labelsize=8)
    for s in ax.spines.values(): s.set_color('#2e3350')

def make_entropy_chart(entropies, confidences):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 3.8), facecolor='#0f1117', sharex=True)
    steps = list(range(1, len(entropies) + 1))
    _dark_ax(ax1)
    ax1.plot(steps, entropies, color='#f78c6c', linewidth=1.5)
    ax1.fill_between(steps, entropies, alpha=0.15, color='#f78c6c')
    if entropies:
        avg = sum(entropies) / len(entropies)
        ax1.axhline(avg, color='#8892b0', linestyle='--', linewidth=0.8)
        ax1.text(len(steps), avg, f' avg {avg:.2f}', color='#8892b0', fontsize=7, va='center')
    ax1.set_ylabel('Uncertainty\n(bits)', color='#8892b0', fontsize=8)
    ax1.set_ylim(0, 5.5)
    ax1.set_title('Model confidence during generation', color='#e2e8f0', fontsize=9, pad=6)
    _dark_ax(ax2)
    cp = [c * 100 for c in confidences]
    ax2.plot(steps, cp, color='#4ade80', linewidth=1.5)
    ax2.fill_between(steps, cp, alpha=0.15, color='#4ade80')
    ax2.set_ylabel('Confidence\n(%)', color='#8892b0', fontsize=8)
    ax2.set_xlabel('Character number', color='#8892b0', fontsize=8)
    ax2.set_ylim(0, 105)
    fig.tight_layout(pad=1.5)
    return fig

def make_score_chart(chars, confidences):
    fig, ax = plt.subplots(figsize=(8, 2.8), facecolor='#0f1117')
    _dark_ax(ax)
    steps  = list(range(1, len(confidences) + 1))
    colors = [conf_to_hex(c) for c in confidences]
    ax.bar(steps, [c * 100 for c in confidences], color=colors, alpha=0.85, width=0.85)
    if confidences:
        avg = sum(confidences) / len(confidences) * 100
        ax.axhline(avg, color='#8892b0', linestyle='--', linewidth=0.9)
        ax.text(len(steps) + 0.5, avg, f' avg {avg:.1f}%', color='#8892b0', fontsize=7, va='center')
    ax.set_ylabel('Confidence %', color='#8892b0', fontsize=8)
    ax.set_xlabel('Character position', color='#8892b0', fontsize=8)
    ax.set_title('How well the model predicted each character  (green=confident · red=surprised)',
                 color='#e2e8f0', fontsize=9)
    ax.set_ylim(0, 108)
    fig.tight_layout(pad=1.2)
    return fig

def make_loss_plot(steps, train_l, val_l):
    fig, ax = plt.subplots(figsize=(6, 3), facecolor='#0f1117')
    _dark_ax(ax)
    ax.plot(steps, train_l, label="train loss", color="#4f8ef7", linewidth=1.5)
    ax.plot(steps, val_l,   label="val loss",   color="#f7804f", linewidth=1.5)
    ax.set_xlabel("Training step", color='#8892b0', fontsize=8)
    ax.set_ylabel("Loss", color='#8892b0', fontsize=8)
    ax.set_title("Training loss — lower is better", color='#e2e8f0', fontsize=9)
    ax.legend(facecolor='#1a1d27', edgecolor='#2e3350', labelcolor='#e2e8f0', fontsize=8)
    ax.grid(True, alpha=0.15, color='#2e3350')
    fig.tight_layout(pad=1.2)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# Verbatim detection
# ══════════════════════════════════════════════════════════════════════════════
def compute_verbatim_mask(text):
    """Return list of bools: True = this character is part of a training n-gram."""
    mask = [False] * len(text)
    for i in range(len(text) - NGRAM_N + 1):
        if text[i:i + NGRAM_N] in _ngrams:
            for j in range(i, i + NGRAM_N):
                if j < len(mask):
                    mask[j] = True
    return mask

# ══════════════════════════════════════════════════════════════════════════════
# Attention extraction
# ══════════════════════════════════════════════════════════════════════════════
def capture_attention(m, context_str):
    """Run one forward pass and return list of (n_head, T, T) attention tensors."""
    tokens  = encode(context_str)
    if not tokens:
        return [], []
    tokens  = tokens[-BLOCK_SIZE:]
    idx_in  = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    for block in m.blocks:
        block.attn._capture = True
    with torch.no_grad():
        m(idx_in)
    for block in m.blocks:
        block.attn._capture = False
    return tokens, [block.attn._last_attn for block in m.blocks]   # list of (n_head, T, T)

def make_attention_strip_html(chars, avg_attn_last_row):
    """Horizontal bar showing which chars the model focused on for the next token."""
    max_a = max(avg_attn_last_row) if max(avg_attn_last_row) > 0 else 1
    html  = ('<div style="font-family:\'SF Mono\',monospace;padding:14px;'
             'background:#0f1117;border-radius:8px;">')
    html += '<div style="color:#8892b0;font-size:11px;margin-bottom:10px;">When deciding the NEXT character, the model focused on:</div>'
    html += '<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:flex-end;">'
    # Sort by attention descending for display
    pairs = sorted(zip(avg_attn_last_row, chars), reverse=True)
    for attn, ch in pairs:
        pct   = attn / max_a
        alpha = max(0.15, pct)
        col   = conf_to_hex(pct)
        label = ch.replace('\n','↵').replace(' ','·').replace('&','&amp;').replace('<','&lt;')
        bar_h = max(4, int(pct * 60))
        html += (f'<div style="display:flex;flex-direction:column;align-items:center;gap:2px">'
                 f'<div style="font-size:9px;color:#8892b0">{attn*100:.0f}%</div>'
                 f'<div style="width:26px;height:{bar_h}px;background:{col};'
                 f'opacity:{alpha:.2f};border-radius:3px 3px 0 0"></div>'
                 f'<div style="font-size:11px;color:#e2e8f0;font-weight:600">{label}</div>'
                 f'</div>')
    html += '</div></div>'
    return html

def make_attention_chart(chars, attn_layers):
    """2×2 grid of attention heatmaps, one per transformer layer."""
    n  = len(chars)
    labels = [c.replace('\n','↵').replace(' ','·') for c in chars]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), facecolor='#0f1117')
    layer_names = ["Layer 1\n(first look)", "Layer 2\n(refining)",
                   "Layer 3\n(deeper pattern)", "Layer 4\n(final decision)"]
    for i, (ax, attn, name) in enumerate(zip(axes.flat, attn_layers, layer_names)):
        _dark_ax(ax)
        avg = attn.mean(0).numpy()  # average across heads: (T, T)
        im  = ax.imshow(avg, cmap='Blues', aspect='auto', vmin=0)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=6, rotation=90)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(name, color='#e2e8f0', fontsize=8, pad=4)
        ax.set_xlabel('Characters it looked at', color='#8892b0', fontsize=7)
        ax.set_ylabel('At each position', color='#8892b0', fontsize=7)
    fig.suptitle('Attention patterns — brighter = the model focused more here',
                 color='#e2e8f0', fontsize=9, y=1.01)
    fig.tight_layout(pad=1.5)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# Core generation step
# ══════════════════════════════════════════════════════════════════════════════
def _gen_step(idx, temp, topk):
    idx_cond     = idx[:, -BLOCK_SIZE:]
    logits       = _model(idx_cond)[:, -1, :]
    probs        = F.softmax(logits / max(temp, 1e-6), dim=-1).squeeze(0)
    H            = entropy_bits(probs)
    top_p, top_i = probs.topk(topk)
    top_p        = top_p.cpu().tolist()
    top_i        = top_i.cpu().tolist()
    chosen_id    = torch.multinomial(probs, 1).item()
    chosen_prob  = probs[chosen_id].item()
    rank         = int((probs > probs[chosen_id]).sum().item()) + 1
    return top_i, top_p, chosen_id, chosen_prob, rank, H

def _quick_generate(m, prompt_tokens, length, temp):
    """Generate `length` chars from m without verbose logging."""
    idx  = torch.tensor([prompt_tokens or [0]], dtype=torch.long, device=DEVICE)
    chars = []; confs = []
    with torch.no_grad():
        for _ in range(length):
            probs    = F.softmax(m(idx[:, -BLOCK_SIZE:])[:, -1, :] / max(temp, 1e-6), dim=-1).squeeze(0)
            cid      = torch.multinomial(probs, 1).item()
            chars.append(itos[cid])
            confs.append(probs[cid].item())
            idx = torch.cat([idx, torch.tensor([[cid]], device=DEVICE)], dim=1)
    return chars, confs

# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(m, step):
    path = os.path.join(CKPT_DIR, f"step_{step:05d}.pt")
    torch.save(m.state_dict(), path)

def list_checkpoints():
    files = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith(".pt") and f.startswith("step_"))
    return [os.path.join(CKPT_DIR, f) for f in files]

def load_checkpoint_model(path):
    m = GPT().to(DEVICE)
    try:
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    except Exception as e:
        print(f"Warning: could not load checkpoint {path}: {e}")
        return m
    m.eval()
    return m

# ── Quality + history helpers ─────────────────────────────────────────────────
def _repeat_ratio(text, n=3):
    if len(text) < n + 1:
        return 0.0
    grams = [text[i:i+n] for i in range(len(text) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))

def assess_generation_quality(text, confidences, temp):
    if not confidences:
        avg_conf = 0.0
    else:
        avg_conf = sum(confidences) / len(confidences)
    repeat = _repeat_ratio(text, n=3)
    if avg_conf >= 0.35 and repeat <= 0.28:
        verdict = "Readable"
        verdict_col = "#4ade80"
    elif repeat > 0.40:
        verdict = "Repetitive"
        verdict_col = "#fbbf24"
    elif avg_conf < 0.16:
        verdict = "Too random"
        verdict_col = "#f97316"
    else:
        verdict = "Needs tuning"
        verdict_col = "#fbbf24"
    tips = []
    fix_action = "none"
    if repeat > 0.40:
        tips.append("Try increasing Creativity by +10 to +20 points.")
        fix_action = "increase_creativity"
    if avg_conf < 0.16:
        tips.append("Try reducing Creativity by 10 points or reducing output length.")
        if fix_action == "none":
            fix_action = "decrease_creativity"
    if temp > 1.4 and avg_conf < 0.20:
        tips.append("Use a lower temperature for more coherent output.")
        if fix_action == "none":
            fix_action = "decrease_creativity"
    if not tips:
        tips.append("Quality looks stable — try a new prompt variation.")
    if verdict == "Needs tuning" and fix_action == "none":
        fix_action = "shorten_length"
    return {
        "verdict": verdict,
        "verdict_col": verdict_col,
        "avg_conf": avg_conf,
        "repeat_ratio": repeat,
        "tips": tips,
        "temp": float(temp),
        "fix_action": fix_action,
    }

def make_quality_scorecard_html(assessment):
    return (
        '<div style="margin-top:10px;background:#1a1d27;border:1px solid #2e3350;border-radius:10px;padding:12px;">'
        '<div style="font-weight:700;color:#e2e8f0;margin-bottom:6px;">Quality scorecard</div>'
        f'<div style="font-size:.86rem;color:#8892b0">Overall: <strong style="color:{assessment["verdict_col"]}">{assessment["verdict"]}</strong></div>'
        f'<div style="font-size:.82rem;color:#8892b0;margin-top:4px;">Avg confidence: {assessment["avg_conf"]*100:.1f}% · Repetition risk: {assessment["repeat_ratio"]*100:.0f}%</div>'
        f'<div style="font-size:.82rem;color:#8892b0;margin-top:7px;">Guardrail tip: {assessment["tips"][0]}</div>'
        '</div>'
    )

def _history_add(kind, title, settings, summary, output_text, assessment=None, meta=None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    _run_history.append({
        "timestamp": ts,
        "kind": kind,
        "title": title,
        "settings": settings or "",
        "summary": summary or "",
        "output_text": output_text or "",
        "assessment": assessment or {},
        "meta": meta or {},
    })
    if len(_run_history) > _MAX_HISTORY:
        del _run_history[0:len(_run_history) - _MAX_HISTORY]
    _save_history_to_disk()

def _save_history_to_disk():
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, "w") as f:
            json.dump(_run_history[-_MAX_HISTORY:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: could not save run history: {e}")

def _load_history_from_disk():
    global _run_history
    if not os.path.exists(HISTORY_PATH):
        return
    try:
        with open(HISTORY_PATH) as f:
            data = json.load(f)
        if isinstance(data, list):
            cleaned = []
            for rec in data[-_MAX_HISTORY:]:
                if not isinstance(rec, dict):
                    continue
                cleaned.append({
                    "timestamp": str(rec.get("timestamp", "")),
                    "kind": str(rec.get("kind", "")),
                    "title": str(rec.get("title", "")),
                    "settings": str(rec.get("settings", "")),
                    "summary": str(rec.get("summary", "")),
                    "output_text": str(rec.get("output_text", "")),
                    "assessment": rec.get("assessment", {}) if isinstance(rec.get("assessment", {}), dict) else {},
                    "meta": rec.get("meta", {}) if isinstance(rec.get("meta", {}), dict) else {},
                })
            _run_history = cleaned
    except Exception as e:
        print(f"Warning: could not load run history: {e}")

def build_history_timeline_md():
    if not _run_history:
        return "No runs yet. Run Generate or Wizard first."
    lines = []
    for i, rec in enumerate(reversed(_run_history), 1):
        q = rec.get("assessment", {}).get("verdict", "—")
        lines.append(f"{i}. `{rec['timestamp']}` · **{rec['title']}** · {rec['kind']} · quality: {q}")
    return "\n".join(lines)

def build_latest_history_html():
    if not _run_history:
        return "<p style='color:#8892b0;padding:10px'>No runs captured yet.</p>"
    rec = _run_history[-1]
    q = rec.get("assessment", {})
    q_html = ""
    if q:
        q_html = (f"<div style='margin-top:8px;color:#8892b0;font-size:.85rem;'>"
                  f"Quality: <strong style='color:{q.get('verdict_col','#fbbf24')}'>{q.get('verdict','')}</strong> · "
                  f"Avg confidence {q.get('avg_conf',0)*100:.1f}% · Repetition {q.get('repeat_ratio',0)*100:.0f}%</div>")
    output_preview = rec["output_text"][:600].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<div style='background:#1a1d27;border:1px solid #2e3350;border-radius:10px;padding:12px;'>"
        f"<div style='color:#e2e8f0;font-weight:700'>{rec['title']}</div>"
        f"<div style='color:#8892b0;font-size:.82rem;margin-top:4px'>{rec['timestamp']} · {rec['settings']}</div>"
        f"<div style='color:#e2e8f0;font-size:.9rem;margin-top:8px;white-space:pre-wrap'>{rec['summary']}</div>"
        f"{q_html}"
        f"<pre style='margin-top:8px;max-height:200px;overflow:auto;background:#0f1117;border-radius:8px;padding:8px;color:#dbeafe;font-size:.78rem;white-space:pre-wrap'>{output_preview}</pre>"
        "</div>"
    )

def refresh_history_view():
    return build_history_timeline_md(), build_latest_history_html()

def export_latest_report():
    if not _run_history:
        return "⚠️ No runs to export yet."
    rec = _run_history[-1]
    os.makedirs("runs/reports", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join("runs", "reports", f"report_{stamp}.md")
    q = rec.get("assessment", {})
    with open(path, "w") as f:
        f.write("# Run Report\n\n")
        f.write(f"- Timestamp: {rec['timestamp']}\n")
        f.write(f"- Title: {rec['title']}\n")
        f.write(f"- Type: {rec['kind']}\n")
        f.write(f"- Settings: {rec['settings']}\n")
        if q:
            f.write(f"- Quality: {q.get('verdict', '')}\n")
            f.write(f"- Avg confidence: {q.get('avg_conf', 0)*100:.1f}%\n")
            f.write(f"- Repetition risk: {q.get('repeat_ratio', 0)*100:.0f}%\n")
            if q.get("tips"):
                f.write(f"- Guardrail tip: {q['tips'][0]}\n")
        f.write("\n## Summary\n\n")
        f.write(rec["summary"] + "\n\n")
        f.write("## Output Preview\n\n```\n")
        f.write(rec["output_text"][:3000])
        f.write("\n```\n")
    return f"✅ Report exported: {path}"

def apply_audience_mode(mode):
    if mode == "Research":
        msg = "Research view enabled: verbose logs and advanced wizard details are on by default."
        return msg, gr.update(value=True), gr.update(value=True)
    msg = "Executive view enabled: concise outputs and simplified defaults."
    return msg, gr.update(value=False), gr.update(value=False)

def generate_mode_change(mode):
    simple = mode == "Simple"
    return (
        gr.update(visible=not simple),   # ui_topk
        gr.update(visible=not simple),   # ui_verbose
        gr.update(visible=not simple),   # gen_log
        gr.update(visible=not simple),   # gen_verb
        gr.update(visible=True),         # gen_underhood
        gr.update(visible=not simple),   # gen_branch_title
        gr.update(visible=not simple),   # gen_branch_desc
        gr.update(visible=not simple),   # branch_step_in
        gr.update(visible=not simple),   # btn_branch
        gr.update(visible=not simple),   # branch_log
        gr.update(visible=not simple),   # branch_html
        gr.update(visible=not simple),   # branch_chart
        gr.update(visible=not simple),   # btn_peek
        gr.update(visible=not simple),   # peek_html
    )

def get_last_generate_record():
    for rec in reversed(_run_history):
        if rec.get("kind") == "generate":
            return rec
    return None

def make_compare_last_run_html(previous_rec, assessment, length, temp):
    if not previous_rec:
        return ('<div style="margin-top:10px;background:#1a1d27;border:1px solid #2e3350;'
                'border-radius:10px;padding:12px;color:#8892b0;font-size:.83rem;">'
                'Compare with last run: no previous generation found yet.</div>')
    prev_assess = previous_rec.get("assessment", {})
    prev_meta = previous_rec.get("meta", {})
    prev_conf = float(prev_assess.get("avg_conf", 0.0))
    cur_conf = float(assessment.get("avg_conf", 0.0))
    conf_delta = (cur_conf - prev_conf) * 100
    prev_rep = float(prev_assess.get("repeat_ratio", 0.0))
    cur_rep = float(assessment.get("repeat_ratio", 0.0))
    rep_delta = (cur_rep - prev_rep) * 100
    prev_len = int(prev_meta.get("length", 0))
    prev_temp = float(prev_meta.get("temp", 0.0))
    conf_msg = f"{conf_delta:+.1f}% confidence"
    rep_msg = f"{rep_delta:+.0f}% repetition risk"
    direction = "improved" if conf_delta >= 0 and rep_delta <= 0 else "mixed" if abs(conf_delta) < 3 and abs(rep_delta) < 8 else "weaker"
    col = "#4ade80" if direction == "improved" else "#fbbf24" if direction == "mixed" else "#f97316"
    return (
        '<div style="margin-top:10px;background:#1a1d27;border:1px solid #2e3350;border-radius:10px;padding:12px;">'
        '<div style="font-weight:700;color:#e2e8f0;margin-bottom:6px;">Compare with last run</div>'
        f'<div style="color:#8892b0;font-size:.83rem;">Overall change: <strong style="color:{col}">{direction}</strong></div>'
        f'<div style="color:#8892b0;font-size:.83rem;">Delta vs last run: {conf_msg} · {rep_msg}</div>'
        f'<div style="color:#8892b0;font-size:.83rem;">Last: temp {prev_temp:.2f}, length {prev_len} · Now: temp {float(temp):.2f}, length {int(length)}</div>'
        '</div>'
    )

def apply_recommended_fix(temp, length):
    global _last_fix_action
    temp = float(temp); length = int(length)
    if _last_fix_action == "increase_creativity":
        new_temp = min(2.0, temp + 0.20)
        return new_temp, length, "✅ Applied: increased creativity (temperature +0.20)."
    if _last_fix_action == "decrease_creativity":
        new_temp = max(0.1, temp - 0.20)
        return new_temp, length, "✅ Applied: reduced creativity (temperature -0.20)."
    if _last_fix_action == "shorten_length":
        new_length = max(20, length - 30)
        return temp, new_length, "✅ Applied: shortened output length (-30)."
    return temp, length, "ℹ️ No fix recommended yet. Run Generate first."

def train_mode_change(mode):
    simple = mode == "Simple"
    return (
        gr.update(visible=not simple),   # ui_lr
        gr.update(visible=not simple),   # train_log
    )

def train_step_hint(steps):
    steps = int(steps)
    est_min = max(1, int(round((steps / 2000) * 3)))
    if steps <= 700:
        quality = "quick preview quality"
    elif steps <= 2500:
        quality = "balanced quality"
    else:
        quality = "best quality (longer run)"
    return f"Estimated time: ~{est_min} min on CPU · expected result: {quality}."

def apply_train_preset(preset):
    if preset.startswith("Quick"):
        steps = 500
    elif preset.startswith("Balanced"):
        steps = 2000
    else:
        steps = 5000
    return steps, train_step_hint(steps)

def replay_mode_change(mode):
    simple = mode == "Simple"
    return (
        gr.update(visible=not simple),   # replay_log
        gr.update(visible=not simple),   # replay_chart
    )

def replay_preset_hint(preset):
    if preset.startswith("Quick"):
        return "Quick glance: shows first updates fast so you can validate that replay works."
    if preset.startswith("Balanced"):
        return "Balanced view: good default for understanding learning without too much detail."
    return "Full replay: best for deep understanding; may take longer if checkpoints are created."

def apply_replay_preset(preset):
    if preset.startswith("Quick"):
        return "Simple", replay_preset_hint(preset), gr.update(visible=False), gr.update(visible=False)
    if preset.startswith("Balanced"):
        return "Simple", replay_preset_hint(preset), gr.update(visible=False), gr.update(visible=False)
    return "Advanced", replay_preset_hint(preset), gr.update(visible=True), gr.update(visible=True)

def make_under_hood_html(prompt, generated, confidences, entropies, ranks, temp):
    gen_text = "".join(generated)
    prompt_tokens = encode(prompt)
    total_steps = len(generated)
    avg_conf = (sum(confidences) / total_steps) if total_steps else 0.0
    avg_entropy = (sum(entropies) / total_steps) if total_steps else 0.0
    top1_pct = (100 * sum(1 for r in ranks if r == 1) / total_steps) if total_steps else 0.0
    surprise_pct = (100 * sum(1 for r in ranks if r > 3) / total_steps) if total_steps else 0.0
    rep = _repeat_ratio(gen_text, n=3)
    randomness = ("Focused" if temp < 0.7 else "Balanced" if temp <= 1.2 else "Creative")
    context_used = min(BLOCK_SIZE, len(prompt_tokens) + total_steps)
    trunc_note = ""
    if len(prompt_tokens) > BLOCK_SIZE:
        trunc_note = f"Prompt was long, so only the last {BLOCK_SIZE} characters were used as context."
    return (
        '<div style="background:#1a1d27;border:1px solid #2e3350;border-radius:10px;padding:12px;">'
        '<div style="font-weight:700;color:#e2e8f0;margin-bottom:6px;">What happened under the hood</div>'
        f'<div style="color:#8892b0;font-size:.86rem;">Mode: <strong style="color:#e2e8f0">{randomness}</strong> '
        f'(temperature {temp:.2f})</div>'
        f'<div style="color:#8892b0;font-size:.82rem;margin-top:5px;">Decisions made: {total_steps} next-character picks</div>'
        f'<div style="color:#8892b0;font-size:.82rem;">Context window used: {context_used}/{BLOCK_SIZE} characters</div>'
        f'<div style="color:#8892b0;font-size:.82rem;">Top guess chosen: {top1_pct:.0f}% · Surprising picks: {surprise_pct:.0f}%</div>'
        f'<div style="color:#8892b0;font-size:.82rem;">Average confidence: {avg_conf*100:.1f}% · Uncertainty: {avg_entropy:.2f} bits</div>'
        f'<div style="color:#8892b0;font-size:.82rem;">Repetition risk: {rep*100:.0f}%</div>'
        f'<div style="color:#8892b0;font-size:.82rem;margin-top:7px;">{trunc_note}</div>'
        '</div>'
    )

def make_chart_insights_html(entropies, confidences):
    if not entropies or not confidences:
        return ""
    avg_entropy = sum(entropies) / len(entropies)
    avg_conf = sum(confidences) / len(confidences)
    hardest_i = entropies.index(max(entropies)) + 1
    easiest_i = confidences.index(max(confidences)) + 1
    trend = "steadier" if abs(confidences[-1] - confidences[0]) < 0.08 else ("improving" if confidences[-1] > confidences[0] else "dropping")
    return (
        '<div style="margin-top:10px;background:#1a1d27;border:1px solid #2e3350;border-radius:10px;padding:12px;">'
        '<div style="font-weight:700;color:#e2e8f0;margin-bottom:6px;">Explain this chart</div>'
        f'<div style="color:#8892b0;font-size:.83rem;">• Overall confidence was <strong style="color:#e2e8f0">{avg_conf*100:.1f}%</strong>.</div>'
        f'<div style="color:#8892b0;font-size:.83rem;">• Confidence trend looked <strong style="color:#e2e8f0">{trend}</strong> across the run.</div>'
        f'<div style="color:#8892b0;font-size:.83rem;">• Hardest choice happened around character <strong style="color:#e2e8f0">{hardest_i}</strong>.</div>'
        f'<div style="color:#8892b0;font-size:.83rem;">• Smoothest choice happened around character <strong style="color:#e2e8f0">{easiest_i}</strong>.</div>'
        f'<div style="color:#8892b0;font-size:.83rem;">• Average uncertainty: <strong style="color:#e2e8f0">{avg_entropy:.2f} bits</strong>.</div>'
        '</div>'
    )

def make_learning_diary_html(prompt, temp, length, assessment, confidences):
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
    next_try = assessment["tips"][0] if assessment.get("tips") else "Try a different prompt style."
    mood = ("calm" if temp < 0.7 else "balanced" if temp <= 1.2 else "exploratory")
    verdict = assessment.get("verdict", "Needs tuning")
    verdict_col = assessment.get("verdict_col", "#fbbf24")
    return (
        '<div style="margin-top:10px;background:#1a1d27;border:1px solid #2e3350;border-radius:10px;padding:12px;">'
        '<div style="font-weight:700;color:#e2e8f0;margin-bottom:6px;">Learning diary (auto note)</div>'
        f'<div style="color:#8892b0;font-size:.83rem;">Today I generated <strong style="color:#e2e8f0">{length}</strong> characters from a prompt of {len(prompt)} characters.</div>'
        f'<div style="color:#8892b0;font-size:.83rem;">Run mood was <strong style="color:#e2e8f0">{mood}</strong> with temperature {temp:.2f}.</div>'
        f'<div style="color:#8892b0;font-size:.83rem;">Quality ended as <strong style="color:{verdict_col}">{verdict}</strong> at {avg_conf*100:.1f}% average confidence.</div>'
        f'<div style="color:#8892b0;font-size:.83rem;">Next recommended try: <strong style=\"color:#e2e8f0\">{next_try}</strong></div>'
        '</div>'
    )

def peek_next_char(context_str, temp):
    if _model is None:
        return "<p style='color:#f87171;padding:12px'>⚠️ No model loaded. Load or train first.</p>"
    context_str = (context_str or "").strip()
    tokens = encode(context_str) or [0]
    idx = torch.tensor([tokens[-BLOCK_SIZE:]], dtype=torch.long, device=DEVICE)
    _model.eval()
    with torch.no_grad():
        probs = F.softmax(_model(idx)[:, -1, :] / max(float(temp), 1e-6), dim=-1).squeeze(0)
    top_p, top_i = probs.topk(5)
    rows = []
    for rank, (pid, prob) in enumerate(zip(top_i.tolist(), top_p.tolist()), 1):
        rows.append(
            f"<tr><td style='padding:4px 8px;color:#e2e8f0'>#{rank}</td>"
            f"<td style='padding:4px 8px;color:#e2e8f0'>{human_char(itos[pid])}</td>"
            f"<td style='padding:4px 8px;color:#8892b0'>{prob*100:.1f}%</td></tr>"
        )
    return (
        '<div style="background:#1a1d27;border:1px solid #2e3350;border-radius:10px;padding:12px;">'
        '<div style="font-weight:700;color:#e2e8f0;margin-bottom:6px;">Next-character peek</div>'
        f"<div style='color:#8892b0;font-size:.82rem;margin-bottom:6px;'>Based on the current context (last {min(len(tokens), BLOCK_SIZE)} chars):</div>"
        '<table style="border-collapse:collapse;width:100%;font-size:.86rem;">'
        "<thead><tr><th style='text-align:left;color:#8892b0;padding:4px 8px;'>Rank</th>"
        "<th style='text-align:left;color:#8892b0;padding:4px 8px;'>Character</th>"
        "<th style='text-align:left;color:#8892b0;padding:4px 8px;'>Chance</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        '</div>'
    )

_load_history_from_disk()

# ── Wizard helpers ────────────────────────────────────────────────────────────
def _wizard_creativity_to_temp(creativity):
    c = max(0.0, min(float(creativity), 100.0))
    return 0.3 + (c / 100.0) * 1.5   # 0.3 → 1.8

def wizard_step_guide(goal):
    if goal == "Train model":
        return ("**Where to start:** Choose how many steps to train, then click **Start Wizard**.\n\n"
                "**Where to see outputs:** Progress appears in **Step 3**, and the learning chart appears in **Chart / visual**.")
    if goal == "Generate text":
        return ("**Where to start:** Enter a prompt and adjust the **Creativity** slider, then click **Start Wizard**.\n\n"
                "**Where to see outputs:** The text appears in **Main output**. Confidence colors and chart appear below.")
    if goal == "Compare styles":
        return ("**Where to start:** Enter one prompt, then click **Start Wizard**.\n\n"
                "**Where to see outputs:** Side-by-side temperature cards appear in **Main output**.")
    return ("**Where to start:** Enter a short context (or reuse your prompt), then click **Start Wizard**.\n\n"
            "**Where to see outputs:** Focus summary appears in **Main output** and attention heatmap in **Chart / visual**.")

def wizard_goal_change(goal):
    guide = wizard_step_guide(goal)
    if goal == "Train model":
        return (
            guide,
            gr.update(visible=False),  # prompt
            gr.update(visible=False),  # context
            gr.update(visible=False),  # creativity
            gr.update(visible=False),  # length
            gr.update(visible=True),   # steps
            gr.update(visible=True),   # lr
        )
    if goal == "Generate text":
        return (
            guide,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    if goal == "Compare styles":
        return (
            guide,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    return (
        guide,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )

def wizard_next_actions(goal):
    if goal == "Train model":
        return ("### Step 5 — What next?\n"
                "- Go to **Generate text** in this wizard and test a prompt.\n"
                "- Or open **⏪ Replay** to watch learning snapshots.")
    if goal == "Generate text":
        return ("### Step 5 — What next?\n"
                "- Try **Compare styles** for side-by-side outputs.\n"
                "- Use **Advanced details** to inspect confidence and logs.")
    if goal == "Compare styles":
        return ("### Step 5 — What next?\n"
                "- Pick your favorite style, then run **Generate text** for a longer output.\n"
                "- Export a summary from the Generate tab if needed.")
    return ("### Step 5 — What next?\n"
            "- Paste a different context to inspect focus changes.\n"
            "- Use **Generate text** to see whether focused characters improve coherence.")

def _wizard_load_model_if_needed():
    if _model is not None:
        return True, "✅  Model is ready."
    msg, _ = load_model()
    return (_model is not None), msg

def run_wizard(goal, prompt, training_steps, lr_val, creativity, length, context_str, show_advanced, audience_mode):
    prompt = (prompt or "").strip()
    context_str = (context_str or "").strip()
    length = int(length)
    show_advanced = bool(show_advanced) and audience_mode == "Research"

    if goal == "Train model":
        guide = wizard_step_guide(goal)
        for log, status, chart in stream_training(training_steps, lr_val):
            if show_advanced:
                main_html = (f"<div style='background:#0f1117;color:#e2e8f0;padding:12px;border-radius:8px;"
                             f"border:1px solid #2e3350'><strong>Training status:</strong> {status}</div>")
                details = f"<pre style='white-space:pre-wrap;margin:0'>{log}</pre>"
            else:
                last_line = [ln for ln in log.strip().splitlines() if ln][-1] if log.strip() else status
                main_html = (f"<div style='background:#0f1117;color:#e2e8f0;padding:12px;border-radius:8px;"
                             f"border:1px solid #2e3350'><strong>{status}</strong><br>"
                             f"Latest update: {last_line}</div>")
                details = ""
            yield ("Step 3 — Running training",
                   guide,
                   main_html,
                   details,
                   chart,
                   wizard_next_actions(goal))
        return

    ready, model_msg = _wizard_load_model_if_needed()
    if not ready:
        yield ("Step 3 — Could not start",
               wizard_step_guide(goal),
               f"<p style='color:#f87171;padding:12px'>⚠️ {model_msg}</p>",
               "",
               None,
               wizard_next_actions(goal))
        return

    temp = _wizard_creativity_to_temp(creativity)
    guide = wizard_step_guide(goal)
    use_prompt = prompt or "ROMEO:\n"

    if goal == "Generate text":
        final = None
        for item in stream_generate(use_prompt, temp, length, 5, bool(show_advanced)):
            final = item
        if final is None:
            yield ("Step 3 — Could not generate", guide, "", "", None, wizard_next_actions(goal))
            return
        log, html, chart, verbatim_html, hood_html, _ = final
        details = (f"<div style='margin-top:8px;color:#8892b0;font-size:.85rem;white-space:pre-wrap'>{log}</div>"
                   if show_advanced else "")
        yield ("Step 3 — Generation complete",
               guide,
               html,
               verbatim_html + hood_html + details,
               chart,
               wizard_next_actions(goal))
        return

    if goal == "Compare styles":
        final_html = None
        for gallery in compare_temperatures(use_prompt, length):
            final_html = gallery
        if final_html is None:
            final_html = "<p style='color:#f87171;padding:12px'>⚠️ No comparison output produced.</p>"
        yield ("Step 3 — Comparison complete",
               guide,
               final_html,
               (f"<div style='color:#8892b0;font-size:.85rem'>Prompt used: <code>{use_prompt}</code></div>"
                if show_advanced else ""),
               None,
               wizard_next_actions(goal))
        return

    context_used = context_str or use_prompt
    strip_html, chart = show_attention(context_used)
    extra = (f"<div style='color:#8892b0;font-size:.85rem'>Context used: <code>{context_used[-18:]}</code></div>"
             if show_advanced else "")
    yield ("Step 3 — Attention analysis complete",
           guide,
           strip_html,
           extra,
           chart,
           wizard_next_actions(goal))

# ══════════════════════════════════════════════════════════════════════════════
# Gradio handlers
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Load saved model ───────────────────────────────────────────────────────
def load_model():
    global _model
    if not os.path.exists(MODEL_PATH):
        return f"❌  {MODEL_PATH} not found. Train first.", None
    _model = GPT().to(DEVICE)
    try:
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except Exception as e:
        _model = None
        return f"❌  Could not load model: {e}. Try training from scratch.", None
    _model.eval()
    return model_status_str(), None

# ── 2. Training (saves checkpoints) ──────────────────────────────────────────
CKPT_STEPS = {100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000}

def stream_training(n_steps, lr_val):
    global _training_in_progress
    if _training_in_progress:
        yield "⚠️  Training is already running. Wait for it to finish.", model_status_str(), None
        return
    _training_in_progress = True
    n_steps = int(n_steps)
    lr_val  = max(1e-6, min(float(lr_val), 0.1))   # clamp to safe range
    _build_model(lr_val)
    writer   = SummaryWriter(log_dir="runs/tiny_shakespeare")
    steps_l  = []; train_l = []; val_l = []
    header   = f"{'Step':>6}  {'Train':>8}  {'Val':>8}  {'GNorm':>7}  {'Time':>8}\n" + "─"*48 + "\n"
    log      = header
    t0       = time.time()
    yield log, model_status_str(), None

    for step in range(1, n_steps + 1):
        x, y   = get_batch("train")
        logits = _model(x)
        loss   = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        _optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = grad_norm(_model)
        nn.utils.clip_grad_norm_(_model.parameters(), GRAD_CLIP)
        _optimizer.step()

        if step in CKPT_STEPS and step <= n_steps:
            save_checkpoint(_model, step)

        if step % 200 == 0:
            losses = estimate_loss(_model)
            elapsed = time.time() - t0
            log += (f"{step:>6}  {losses['train']:>8.4f}  {losses['val']:>8.4f}"
                    f"  {gn:>7.3f}  {elapsed:>7.1f}s\n")
            steps_l.append(step); train_l.append(losses["train"]); val_l.append(losses["val"])
            writer.add_scalar("loss/train", losses["train"], step)
            writer.add_scalar("loss/val",   losses["val"],   step)
            writer.add_scalar("grad_norm",  gn,              step)
            yield log, f"⏳  Step {step} / {n_steps}", make_loss_plot(steps_l, train_l, val_l)

    writer.close()
    torch.save(_model.state_dict(), MODEL_PATH)
    _model.eval()
    log += f"\n✅  Done!  Saved → {MODEL_PATH}\n"
    log += f"✅  Checkpoints saved to {CKPT_DIR}/  (use Replay tab to watch learning)\n"
    _history_add(
        kind="train",
        title="Training run",
        settings=f"steps={n_steps}, lr={lr_val:.5f}",
        summary="Training finished and model/checkpoints were saved.",
        output_text=log[-1800:],
    )
    _training_in_progress = False
    yield log, model_status_str(), make_loss_plot(steps_l, train_l, val_l)

# ── 3. Generate ───────────────────────────────────────────────────────────────
def stream_generate(prompt, temp, length, topk, verbose):
    global _model, _gen_history, _last_temp, _last_prompt, _last_fix_action

    if _model is None:
        yield "⚠️  No model loaded.", "", None, "", "", ""
        return

    _model.eval()
    tokens  = encode(prompt) or [0]
    idx     = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    length  = int(length); topk = int(topk); temp = float(temp)
    _gen_history = [idx.clone()]; _last_temp = temp; _last_prompt = prompt

    # Immediately clear stale outputs from any previous generation
    yield "", "<p style='color:#8892b0;padding:14px'>⏳  Generating…</p>", None, "", "", ""

    verbose_log = ""; generated = []; entropies = []; confidences = []; ranks = []

    if verbose:
        verbose_log  = f'Generating from: "{prompt}"\n'
        verbose_log += f"Temperature: {temp}  |  {length} characters\n" + "="*54 + "\n"

    with torch.no_grad():
        for step in range(1, length + 1):
            top_i, top_p, chosen_id, chosen_prob, rank, H = _gen_step(idx, temp, topk)
            entropies.append(H); confidences.append(chosen_prob)
            ranks.append(rank); generated.append(itos[chosen_id])
            idx = torch.cat([idx, torch.tensor([[chosen_id]], device=DEVICE)], dim=1)
            _gen_history.append(idx.clone())

            if verbose:
                ctx_raw = decode(idx[0].tolist()[:-1][-20:])
                ctx_disp = ctx_raw.replace('\n','↵').replace(' ','·')
                block  = f"\n{'─'*54}\n  Character {step} of {length}\n{'─'*54}\n"
                block += f"\n  The model just read:  \"{ctx_disp}\"\n"
                block += f"  How sure is it?       {entropy_label(H)}\n"
                block += f"\n  What it was considering:\n\n"
                for pid, pp in zip(top_i, top_p):
                    name = human_char(itos[pid]); bar = "█"*int(pp*22)
                    pct  = f"{pp*100:.1f}%"; tick = "  ✓ picked" if pid==chosen_id else ""
                    block += f"    {name:<16}  {bar:<22}  {pct:>6}{tick}\n"
                if chosen_id not in top_i:
                    block += f"    {human_char(itos[chosen_id]):<16}  {'·'*22}  {chosen_prob*100:.1f}%  ✓ picked\n"
                block += f"\n  ➜  Chose {human_char(itos[chosen_id])}  —  {choice_comment(rank, chosen_prob)}\n"
                verbose_log += block

            chart    = make_entropy_chart(entropies, confidences) if step % 15 == 0 else gr.skip()
            progress = f"Generating… {step} of {length} characters" if not verbose else verbose_log
            yield (progress,
                   make_colored_html(prompt, generated, confidences),
                   chart, "", "", "")

    # Final summary
    avg_conf = sum(confidences) / len(confidences)
    pct_top1 = 100 * sum(1 for r in ranks if r == 1) / len(ranks)
    max_H_i  = entropies.index(max(entropies))
    max_c_i  = confidences.index(max(confidences))
    cf       = ("very confident" if avg_conf > .5 else "moderately confident" if avg_conf > .3
                else "often uncertain" if avg_conf > .15 else "frequently guessing")
    tn       = "focused" if temp < .7 else "creative" if temp > 1.2 else "balanced"
    summary  = f"✅  Done! Generated {length} characters at temperature {temp} ({tn} mode).\n\n"
    summary += f"The model was {cf}, choosing its top guess {pct_top1:.0f}% of the time.\n\n"
    summary += f"Most confident moment: character {max_c_i+1} ({human_char(generated[max_c_i])}) — {confidences[max_c_i]*100:.1f}% sure.\n"
    summary += f"Hardest decision: character {max_H_i+1} ({human_char(generated[max_H_i])}) — many characters seemed equally likely.\n\n"
    summary += f"💡 Use Branch below to explore what else could have been written."

    if verbose:
        verbose_log += summary

    gen_text    = "".join(generated)
    v_mask      = compute_verbatim_mask(gen_text)
    v_html      = make_verbatim_html(prompt, list(gen_text), v_mask)
    assessment  = assess_generation_quality(gen_text, confidences, temp)
    prev_gen    = get_last_generate_record()
    _last_fix_action = assessment.get("fix_action", "none")
    hood_html   = (
        make_under_hood_html(prompt, generated, confidences, entropies, ranks, temp)
        + make_chart_insights_html(entropies, confidences)
        + make_learning_diary_html(prompt, temp, length, assessment, confidences)
        + make_compare_last_run_html(prev_gen, assessment, length, temp)
    )
    scorecard   = make_quality_scorecard_html(assessment)
    _history_add(
        kind="generate",
        title="Generate output",
        settings=f"temp={temp:.2f}, length={length}, topk={topk}",
        summary=summary,
        output_text=prompt + gen_text,
        assessment=assessment,
        meta={"length": int(length), "temp": float(temp), "prompt_len": len(prompt)},
    )

    yield (verbose_log if verbose else summary,
           make_colored_html(prompt, generated, confidences),
           make_entropy_chart(entropies, confidences),
           v_html + scorecard,
           hood_html,
           "")

# ── 4. Branch ─────────────────────────────────────────────────────────────────
def branch_from_step(branch_step):
    global _gen_history, _last_temp, _last_prompt, _model
    if _model is None:
        yield "⚠️  No model loaded.", "", None; return
    if not _gen_history:
        yield "⚠️  Generate something first.", "", None; return
    requested   = int(branch_step)
    branch_step = max(1, min(requested, len(_gen_history) - 1))
    idx  = _gen_history[branch_step - 1].clone()
    temp = _last_temp
    ctx  = decode(idx[0].tolist()[-20:]).replace('\n','↵').replace(' ','·')
    log  = f"🌿  Branching from character {branch_step}"
    if requested != branch_step:
        log += f"  (you selected {requested} — clamped to end of generated text)"
    log += "\n"
    log += f"    Context: \"…{ctx}\"\n    Generating 120 new characters...\n" + "─"*50 + "\n\n"
    generated = []; confidences = []; entropies = []
    _model.eval()
    with torch.no_grad():
        for step in range(1, 121):
            top_i, top_p, chosen_id, chosen_prob, rank, H = _gen_step(idx, temp, topk=5)
            generated.append(itos[chosen_id]); confidences.append(chosen_prob); entropies.append(H)
            idx = torch.cat([idx, torch.tensor([[chosen_id]], device=DEVICE)], dim=1)
            if step % 20 == 0 or step == 120:
                yield (log + "".join(generated),
                       make_colored_html("", generated, confidences),
                       make_entropy_chart(entropies, confidences))
    log += "".join(generated) + "\n\n🌿  Branch complete.\n"
    yield log, make_colored_html("", generated, confidences), make_entropy_chart(entropies, confidences)

# ── 5. Score text ─────────────────────────────────────────────────────────────
def score_text(input_text):
    if _model is None: return "⚠️  No model loaded.", None, ""
    if len(input_text.strip()) < 2: return "⚠️  Paste at least 2 characters.", None, ""
    # Cap at 200 chars for speed; re-encode via decode to drop OOV chars and prevent index misalignment
    if len(input_text) > 200:
        input_text = input_text[:200]
    display_text = decode(encode(input_text))   # only chars in Shakespeare vocab
    tokens = encode(display_text)
    if len(tokens) < 2: return "⚠️  No recognisable Shakespeare characters found.", None, ""
    oov_note = (f" (Note: {len(input_text) - len(display_text)} characters outside Shakespeare's alphabet were skipped)"
                if len(input_text) != len(display_text) else "")
    _model.eval()
    per_conf = []; per_rank = []
    with torch.no_grad():
        for i in range(len(tokens) - 1):
            ctx   = tokens[max(0, i+1-BLOCK_SIZE): i+1]
            probs = F.softmax(_model(torch.tensor([ctx], dtype=torch.long, device=DEVICE))[:,-1,:], dim=-1).squeeze(0)
            conf  = probs[tokens[i+1]].item()
            rank  = int((probs > probs[tokens[i+1]]).sum().item()) + 1
            per_conf.append(conf); per_rank.append(rank)
    html = _WRAP
    c0 = display_text[0]; c0s = c0.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    html += f'<span style="opacity:.35" title="First character — context only">{"<br>" if c0=="\n" else ("&nbsp;" if c0==" " else c0s)}</span>'
    for char, conf, rank in zip(display_text[1:], per_conf, per_rank):
        html += _span(char, conf_to_hex(conf), f"{conf*100:.1f}% confident (rank #{rank})")
    html += '</div>' + _KEY
    avg_conf   = sum(per_conf) / len(per_conf)
    perplexity = math.exp(-sum(math.log(max(p,1e-10)) for p in per_conf) / len(per_conf))
    pct_top1   = 100 * sum(1 for r in per_rank if r==1) / len(per_rank)
    if perplexity < 3:    ppl_d = "the model knows this text very well"
    elif perplexity < 6:  ppl_d = "the model has a reasonable grasp of it"
    elif perplexity < 12: ppl_d = "parts of this are unfamiliar to the model"
    else:                 ppl_d = "this text surprises the model a lot"
    indexed   = sorted(enumerate(per_conf), key=lambda x: x[1])
    surprises = [f'position {i+2} ("{human_char(display_text[i+1])}")' for i,_ in indexed[:3]]
    stats  = f"Perplexity: {perplexity:.1f} out of max ~{VOCAB_SIZE}  —  {ppl_d}\n"
    stats += f"Avg confidence: {avg_conf*100:.1f}%  |  Predicted #1 guess: {pct_top1:.0f}% of the time\n"
    stats += f"Most surprising: {', '.join(surprises)}{oov_note}"
    return html, make_score_chart(list(display_text[1:]), per_conf), stats

# ── 6. Temperature gallery ────────────────────────────────────────────────────
_TEMP_SPECS = [
    (0.3, "0.3 — Focused",   "Very predictable, often repetitive",             "#4ade80"),
    (0.6, "0.6 — Careful",   "Conservative, tends to follow common patterns",  "#86efac"),
    (1.0, "1.0 — Balanced",  "Default — creative yet coherent",                "#fbbf24"),
    (1.4, "1.4 — Creative",  "More adventurous, occasional surprises",         "#f97316"),
    (1.8, "1.8 — Wild",      "Unpredictable, chaotic output",                  "#f87171"),
]

def _gallery_html(prompt, results):
    html = ('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));'
            'gap:14px;padding:4px;">')
    for r in results:
        col   = r['col']
        chtml = make_colored_html("", list(r['text']), r['confs'])
        html += (f'<div style="background:#1a1d27;border:1px solid #2e3350;border-radius:10px;overflow:hidden;">'
                 f'<div style="background:{col}22;border-bottom:2px solid {col};padding:10px 14px;">'
                 f'<div style="font-weight:700;color:#fff;font-size:.92rem;">🌡️ {r["label"]}</div>'
                 f'<div style="color:#8892b0;font-size:.78rem;margin-top:2px;">{r["desc"]}</div>'
                 f'</div>'
                 f'<div style="padding:10px 14px;max-height:220px;overflow-y:auto;">{chtml}</div>'
                 f'<div style="padding:7px 14px;border-top:1px solid #2e3350;color:#8892b0;font-size:.76rem;">'
                 f'Avg confidence: <strong style="color:{col}">{r["avg_conf"]:.1f}%</strong></div>'
                 f'</div>')
    html += '</div>'
    html += ('<div style="color:#8892b0;font-size:.78rem;padding:6px 4px;margin-top:4px;">'
             '💡 Lower confidence at higher temperatures doesn\'t mean worse writing — '
             'it means the model is exploring more options. Creative writing often benefits from lower confidence.</div>')
    return html

def compare_temperatures(prompt, length):
    if _model is None:
        yield "<p style='color:#f87171;padding:16px'>⚠️  No model loaded. Go to Train tab first.</p>"
        return
    if not prompt.strip():
        yield "<p style='color:#f87171;padding:16px'>⚠️  Please enter a prompt first.</p>"
        return
    _model.eval()
    length  = int(length)
    results = []
    tokens  = encode(prompt) or [0]
    for temp, label, desc, col in _TEMP_SPECS:
        chars, confs = _quick_generate(_model, tokens, length, temp)
        if not confs:
            continue
        results.append({
            'label': label, 'desc': desc, 'col': col,
            'text': "".join(chars), 'confs': confs,
            'avg_conf': sum(confs) / len(confs) * 100,
        })
        yield _gallery_html(prompt, results)
    if results:
        _history_add(
            kind="compare",
            title="Temperature comparison",
            settings=f"length={length}, prompt_len={len(prompt)}",
            summary=f"Compared {len(results)} temperature settings.",
            output_text=prompt,
            assessment={"verdict": "Compared", "verdict_col": "#60a5fa"},
        )

# ── 7. Training replay ────────────────────────────────────────────────────────
def _replay_perplexity_chart(steps_done, ppls):
    fig, ax = plt.subplots(figsize=(7, 3), facecolor='#0f1117')
    _dark_ax(ax)
    ax.plot(steps_done, ppls, color='#7c83fd', linewidth=2, marker='o', markersize=5)
    ax.fill_between(steps_done, ppls, alpha=0.15, color='#7c83fd')
    ax.set_ylabel('Perplexity\n(lower = better)', color='#8892b0', fontsize=8)
    ax.set_xlabel('Training step', color='#8892b0', fontsize=8)
    ax.set_title('How the model improved during training  —  lower = smarter',
                 color='#e2e8f0', fontsize=9)
    fig.tight_layout(pad=1.5)
    return fig

def _replay_samples_html(records):
    html = ('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));'
            'gap:12px;padding:4px;">')
    for rec in records:
        stage = rec['stage']
        ppl   = rec['ppl']
        chtml = make_colored_html("", list(rec['text']), rec['confs'])
        if ppl < 5:    badge_col = "#4ade80"
        elif ppl < 10: badge_col = "#fbbf24"
        else:          badge_col = "#f87171"
        html += (f'<div style="background:#1a1d27;border:1px solid #2e3350;border-radius:10px;overflow:hidden;">'
                 f'<div style="padding:9px 14px;background:#22263a;border-bottom:1px solid #2e3350;">'
                 f'<span style="font-weight:700;color:#fff;font-size:.88rem;">{stage}</span>'
                 f'  <span style="background:{badge_col}33;color:{badge_col};font-size:.74rem;'
                 f'padding:2px 8px;border-radius:10px;margin-left:8px">perplexity {ppl:.1f}</span>'
                 f'</div>'
                 f'<div style="padding:10px 14px;max-height:160px;overflow-y:auto;font-size:.82rem;">{chtml}</div>'
                 f'</div>')
    html += '</div>'
    return html

def run_replay(gen_prompt):
    ckpts = list_checkpoints()

    if not ckpts:
        yield ("No checkpoints found. Running a quick training demo (1 000 steps)…\n"
               "This takes about 2 minutes. Please wait.", None, "")

        # Quick demo: train a fresh model, save checkpoints every 100 steps
        demo_m   = GPT().to(DEVICE)
        demo_opt = torch.optim.AdamW(demo_m.parameters(), lr=3e-4)
        demo_m.train()
        for step in range(1, 1001):
            x, y = get_batch("train")
            loss = F.cross_entropy(demo_m(x).view(-1, VOCAB_SIZE), y.view(-1))
            demo_opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(demo_m.parameters(), GRAD_CLIP); demo_opt.step()
            if step % 100 == 0 or step == 1:
                save_checkpoint(demo_m, step)
                yield (f"Quick demo training: step {step}/1000…", None, "")
        ckpts = list_checkpoints()

    gen_tokens = encode(gen_prompt) or [0]
    steps_done = []; ppls = []; records = []

    for ckpt_path in ckpts:
        step_num = int(os.path.basename(ckpt_path).replace("step_","").replace(".pt",""))
        m        = load_checkpoint_model(ckpt_path)
        ppl      = compute_perplexity(m, _REPLAY_TEST)
        chars, confs = _quick_generate(m, gen_tokens, 80, 1.0)

        stage = (f"Step {step_num}  —  "
                 + ("untrained (random)" if step_num <= 1 else
                    "early learning" if step_num < 300 else
                    "getting there" if step_num < 700 else "well trained"))

        steps_done.append(step_num); ppls.append(ppl)
        records.append({'stage': stage, 'ppl': ppl, 'text': "".join(chars), 'confs': confs})

        log = (f"Processed {len(records)}/{len(ckpts)} checkpoints…\n"
               f"Step {step_num}: perplexity = {ppl:.1f}")
        yield log, _replay_perplexity_chart(steps_done, ppls), _replay_samples_html(records)

    if not ppls:
        yield "❌  No checkpoints could be loaded.", None, ""
        return
    summary = (f"✅  Replay complete!  {len(records)} snapshots processed.\n\n"
               f"Perplexity went from {ppls[0]:.1f} → {ppls[-1]:.1f}  "
               f"({100*(1 - ppls[-1]/ppls[0]):.0f}% improvement).\n\n"
               f"Lower perplexity = the model better predicts the text = it has learned more.")
    yield summary, _replay_perplexity_chart(steps_done, ppls), _replay_samples_html(records)

# ── 8. Attention heatmap ──────────────────────────────────────────────────────
def show_attention(context_str):
    if _model is None:
        return "<p style='color:#f87171;padding:16px'>⚠️  No model loaded.</p>", None

    _model.eval()
    if not context_str.strip():
        return "<p style='color:#f87171;padding:16px'>⚠️  Please enter a context string.</p>", None
    context_str = context_str[-18:]   # keep last 18 chars for readability
    tokens, attn_layers = capture_attention(_model, context_str)

    if not tokens or any(a is None for a in attn_layers):
        return "<p style='color:#f87171;padding:16px'>⚠️  Could not extract attention. Use a longer prompt.</p>", None

    chars = [itos[t] for t in tokens]
    T     = len(chars)

    # Average attention across all layers and heads: (T, T)
    avg_all = sum(a.mean(0) for a in attn_layers) / len(attn_layers)  # (T, T)

    # Last row = how much each previous char was attended to when predicting NEXT token
    last_row = avg_all[-1].numpy()   # shape (T,)
    last_row = last_row / (last_row.sum() + 1e-8)

    # Natural language summary: top 3 most attended chars
    top_idx  = sorted(range(T), key=lambda i: last_row[i], reverse=True)[:3]
    top_desc = [f'"{human_char(chars[i])}" ({last_row[i]*100:.0f}%)' for i in top_idx]
    summary  = (f'When deciding the next character, the model focused most on: '
                f'{", ".join(top_desc)}.')

    strip_html = make_attention_strip_html(chars, last_row.tolist())
    # Add the summary below the strip
    strip_html += (f'<div style="padding:10px 14px;color:#8892b0;font-size:.88rem;">'
                   f'💡 {summary}</div>')

    chart = make_attention_chart(chars, attn_layers)
    _history_add(
        kind="attention",
        title="Attention analysis",
        settings=f"context_len={len(context_str)}",
        summary=summary,
        output_text=context_str,
    )
    return strip_html, chart

# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(title="Shakespeare GPT") as demo:

    gr.Markdown(
        "# 🎭 Tiny Shakespeare GPT\n"
        f"Character-level GPT  |  {VOCAB_SIZE}-char vocab  |  "
        f"{N_LAYER} layers · {N_HEAD} heads · {N_EMBD}-dim  |  device: **{DEVICE}**"
    )
    status_bar = gr.Textbox(label="Model Status", value=model_status_str(), interactive=False)
    audience_mode = gr.Radio(
        choices=["Executive", "Research"],
        value="Executive",
        label="Audience Mode"
    )
    audience_note = gr.Markdown("Executive view enabled: concise outputs and simplified defaults.")

    # ── Tab 0: Wizard ─────────────────────────────────────────────────────────
    with gr.Tab("🧭  Wizard"):
        gr.Markdown(
            "### Start here\n"
            "Pick a goal, fill the minimum inputs, then click **Start Wizard**.\n"
            "Outputs always appear in the same place: **Step 3 status**, **Main output**, then **Chart / visual**."
        )
        wiz_goal = gr.Radio(
            choices=["Generate text", "Train model", "Compare styles", "Understand model"],
            value="Generate text",
            label="Step 1 — What do you want to do?"
        )
        wiz_where = gr.Markdown(value=wizard_step_guide("Generate text"))
        with gr.Row():
            with gr.Column(scale=1):
                wiz_prompt = gr.Textbox(
                    label="Step 2 — Prompt",
                    placeholder="ROMEO:\n",
                    value="ROMEO:\n",
                    lines=4
                )
                wiz_context = gr.Textbox(
                    label="Context for focus view (optional)",
                    placeholder="Used only for Understand model",
                    lines=2,
                    visible=False
                )
                wiz_creativity = gr.Slider(
                    0, 100, value=50, step=5,
                    label="Creativity (0 = focused, 100 = wild)"
                )
                wiz_length = gr.Slider(
                    20, 300, value=120, step=10,
                    label="Output length"
                )
                wiz_steps = gr.Slider(
                    100, 3000, value=600, step=100,
                    label="Training steps (only for Train model)",
                    visible=False
                )
                wiz_lr = gr.Slider(
                    1e-5, 1e-3, value=3e-4, step=1e-5,
                    label="Learning rate (only for Train model)",
                    visible=False
                )
                wiz_advanced = gr.Checkbox(value=False, label="Advanced details")
                wiz_start = gr.Button("▶️  Start guided run", variant="primary")
                wiz_export = gr.Button("🧾 Save latest report", variant="secondary")
            with gr.Column(scale=2):
                wiz_status = gr.Textbox(label="Step 3 — Progress / status", lines=4, interactive=False)
                wiz_main = gr.HTML(label="Step 4 — Main output")
                wiz_extra = gr.HTML(label="More details")
                wiz_chart = gr.Plot(label="Chart / visual")
                wiz_next = gr.Markdown("### Step 5 — What next?\n- Run the wizard to see recommended next actions.")
                wiz_export_status = gr.Textbox(label="Report status", lines=2, interactive=False)

    # ── Tab 0.5: Run History ──────────────────────────────────────────────────
    with gr.Tab("📚  History"):
        gr.Markdown("Timeline of recent runs, with quality scorecard and quick export.")
        with gr.Row():
            with gr.Column(scale=1):
                hist_refresh = gr.Button("🔄 Reload history", variant="secondary")
                hist_export = gr.Button("🧾 Save latest report", variant="primary")
                hist_export_status = gr.Textbox(label="Export status", lines=2, interactive=False)
                hist_timeline = gr.Markdown(value=build_history_timeline_md())
            with gr.Column(scale=2):
                hist_latest = gr.HTML(label="Latest run details", value=build_latest_history_html())

    # ── Tab 1: Train ──────────────────────────────────────────────────────────
    with gr.Tab("🏋️  Train"):
        gr.Markdown("**What this tab does:** teaches the model from scratch or loads your saved model.\n\n"
                    "**Where outputs appear:** learning chart on the right, detailed progress below.")
        with gr.Row():
            with gr.Column(scale=1):
                train_mode = gr.Radio(
                    choices=["Simple", "Advanced"],
                    value="Simple",
                    label="View"
                )
                train_preset = gr.Radio(
                    choices=["Quick preview", "Balanced (recommended)", "Best quality"],
                    value="Balanced (recommended)",
                    label="Training goal",
                    info="Pick a goal and the app sets training steps for you."
                )
                ui_steps = gr.Slider(100, 5000, value=2000, step=100, label="Training Steps",
                                     info="How long to train. More steps usually improves output quality.")
                train_hint = gr.Markdown(train_step_hint(2000))
                ui_lr    = gr.Slider(1e-5, 1e-3, value=3e-4, step=1e-5,
                                     label="Learning Rate",
                                     info="How fast learning updates happen each step. Keep default unless testing.",
                                     visible=False)
                with gr.Row():
                    btn_load  = gr.Button("📂  Use saved model", variant="secondary")
                    btn_train = gr.Button("🚀  Start training", variant="primary")
                gr.Markdown("Use **Use saved model** if you trained before. Use **Start training** to start fresh.")
            with gr.Column(scale=2):
                train_loss_plot = gr.Plot(label="Learning Progress")
        train_log = gr.Textbox(label="Detailed progress (Advanced)", lines=18, max_lines=40,
                               elem_classes=["monospace"], interactive=False, visible=False)

    # ── Tab 2: Generate ───────────────────────────────────────────────────────
    with gr.Tab("🎭  Generate"):
        gr.Markdown("**What this tab does:** writes new text from your prompt.\n\n"
                    "**Where outputs appear:** generated text and confidence chart on the right.")
        with gr.Row():
            with gr.Column(scale=1):
                gen_mode = gr.Radio(
                    choices=["Simple", "Advanced"],
                    value="Simple",
                    label="View"
                )
                ui_prompt  = gr.Textbox(label="Prompt", placeholder="ROMEO:\n", lines=4,
                                        info="Starting text the model continues from.")
                ui_temp    = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature",
                                       info="Low = safer text. High = more creative/risky text.")
                ui_length  = gr.Slider(20, 400, value=150, step=10, label="Characters to generate",
                                       info="How long you want the output to be.")
                with gr.Accordion("Advanced generation controls", open=False):
                    ui_topk    = gr.Slider(1, 10, value=5, step=1,
                                           label="Options shown in step-by-step log",
                                           info="How many options to display in advanced logs. Does not change final output.",
                                           visible=False)
                    ui_verbose = gr.Checkbox(value=False, label="Verbose log", visible=False)
                btn_gen    = gr.Button("✨  Create text", variant="primary")
                btn_apply_fix = gr.Button("🛠️ Auto-fix settings", variant="secondary")
                fix_status = gr.Textbox(label="Fix status", lines=2, interactive=False)
            with gr.Column(scale=2):
                gen_log   = gr.Textbox(label="Step-by-step log", lines=18, max_lines=60,
                                       elem_classes=["monospace"], interactive=False, visible=False)
                gen_html  = gr.HTML(label="Generated text  (colour = confidence)")
                gen_chart = gr.Plot(label="Confidence Trend")
                with gr.Accordion("🔍 Verbatim check — is this copied or created?", open=False):
                    gen_verb = gr.HTML(visible=False)
                with gr.Accordion("🧠 Under the hood — simple explanation", open=True):
                    gen_underhood = gr.HTML()
                with gr.Accordion("🔎 Peek next character", open=False):
                    gr.Markdown("See the model's top 5 next-character guesses for your current prompt.")
                    btn_peek = gr.Button("🔎 Show next guesses", variant="secondary", visible=False)
                    peek_html = gr.HTML(visible=False)
        gen_branch_title = gr.Markdown("---\n### 🌿 Branch — explore an alternative path", visible=False)
        gen_branch_desc = gr.Markdown("Pick any character number and re-roll from that point. Same context, different outcome.", visible=False)
        with gr.Row():
            branch_step_in = gr.Slider(1, 400, value=10, step=1, label="Branch from character number", visible=False)
            btn_branch     = gr.Button("🌿  Try alternate version", variant="secondary", visible=False)
        with gr.Row():
            branch_log   = gr.Textbox(label="Branch log", lines=5, interactive=False,
                                      elem_classes=["monospace"], visible=False)
            branch_html  = gr.HTML(label="Branch text", visible=False)
        branch_chart = gr.Plot(label="Branch confidence chart", visible=False)

    # ── Tab 3: Inspect (Score + Attention) ───────────────────────────────────
    with gr.Tab("🔬  Inspect"):
        gr.Markdown("**What this tab does:** helps you inspect model quality and model focus in one place.\n\n"
                    "**Where outputs appear:** charts and visuals on the right.")
        with gr.Tab("🔬  Score Text"):
            gr.Markdown("Checks how familiar the model is with your text.")
            with gr.Row():
                with gr.Column(scale=1):
                    score_input = gr.Textbox(label="Paste text here",
                        placeholder="To be, or not to be, that is the question.", lines=6,
                        info="Paste text to see where the model felt confident or surprised.")
                    btn_score   = gr.Button("🔬  Check this text", variant="primary")
                    score_stats = gr.Textbox(label="Simple summary",
                                             lines=4, interactive=False)
                with gr.Column(scale=2):
                    score_html  = gr.HTML(label="Character confidence")
                    score_chart = gr.Plot(label="Confidence per character")

        with gr.Tab("🔭  Attention Focus"):
            gr.Markdown("Shows which earlier characters the model looked at before choosing the next one.")
            with gr.Row():
                with gr.Column(scale=1):
                    attn_ctx   = gr.Textbox(label="Context (up to 18 characters)",
                                            placeholder="ROMEO:\nWhat light", lines=3,
                                            info="Enter a short text snippet to inspect model focus.")
                    btn_attn   = gr.Button("🔭  Show model focus", variant="primary")
                    gr.Markdown("*Tip: try entering the start of a name like `ROMEO` — "
                                "the model should show that it focuses on the earlier letters when deciding what comes next.*")
                with gr.Column(scale=2):
                    attn_strip = gr.HTML(label="Which characters the model focused on")
                    attn_chart = gr.Plot(label="Per-layer attention heatmaps")

    # ── Tab 4: Temperature Gallery ────────────────────────────────────────────
    with gr.Tab("🌡️  Temperature"):
        gr.Markdown(
            "**What this tab does:** compares the same prompt at different creativity levels.\n\n"
            "**Where outputs appear:** side-by-side cards on the right."
        )
        with gr.Row():
            with gr.Column(scale=1):
                temp_prompt = gr.Textbox(label="Prompt", placeholder="ROMEO:\n", lines=3,
                                         value="ROMEO:\n",
                                         info="Same prompt is used for all temperature cards.")
                temp_length = gr.Slider(40, 150, value=80, step=10,
                                        label="Characters per temperature",
                                        info="How long each comparison output should be.")
                btn_compare = gr.Button("🌡️  Compare creativity levels", variant="primary")
            with gr.Column(scale=3):
                gallery_html = gr.HTML(label="Five temperatures — generated simultaneously")

    # ── Tab 5: Training Replay ────────────────────────────────────────────────
    with gr.Tab("⏪  Replay"):
        gr.Markdown(
            "**What this tab does:** shows how model output improves across training snapshots.\n\n"
            "**Where outputs appear:** progress updates on the left, improvement chart and examples on the right."
        )
        with gr.Row():
            with gr.Column(scale=1):
                replay_mode = gr.Radio(
                    choices=["Simple", "Advanced"],
                    value="Simple",
                    label="View"
                )
                replay_preset = gr.Radio(
                    choices=["Quick glance", "Balanced (recommended)", "Full replay"],
                    value="Balanced (recommended)",
                    label="Replay goal",
                    info="Choose how deep you want the replay experience."
                )
                replay_hint = gr.Markdown(replay_preset_hint("Balanced (recommended)"))
                replay_prompt = gr.Textbox(label="Prompt to generate at each step",
                                           value="ROMEO:\n", lines=3,
                                           info="This same prompt is used for every snapshot.")
                btn_replay    = gr.Button("▶️  Show learning replay", variant="primary")
                replay_log    = gr.Textbox(label="Progress", lines=5, interactive=False, visible=False)
            with gr.Column(scale=2):
                replay_chart  = gr.Plot(label="Model confusion over training (lower is better)", visible=False)
        replay_grid = gr.HTML(label="Generated text at each training checkpoint")

    # ── Wire up events ────────────────────────────────────────────────────────
    btn_load.click(fn=load_model, outputs=[status_bar, train_loss_plot])
    btn_train.click(fn=stream_training, inputs=[ui_steps, ui_lr],
                    outputs=[train_log, status_bar, train_loss_plot])
    train_mode.change(fn=train_mode_change, inputs=[train_mode], outputs=[ui_lr, train_log])
    train_preset.change(fn=apply_train_preset, inputs=[train_preset], outputs=[ui_steps, train_hint])
    ui_steps.change(fn=train_step_hint, inputs=[ui_steps], outputs=[train_hint])
    btn_gen.click(fn=stream_generate,
                  inputs=[ui_prompt, ui_temp, ui_length, ui_topk, ui_verbose],
                  outputs=[gen_log, gen_html, gen_chart, gen_verb, gen_underhood, branch_log])
    ui_prompt.submit(fn=stream_generate,
                     inputs=[ui_prompt, ui_temp, ui_length, ui_topk, ui_verbose],
                     outputs=[gen_log, gen_html, gen_chart, gen_verb, gen_underhood, branch_log])
    btn_peek.click(fn=peek_next_char, inputs=[ui_prompt, ui_temp], outputs=[peek_html])
    btn_apply_fix.click(fn=apply_recommended_fix, inputs=[ui_temp, ui_length], outputs=[ui_temp, ui_length, fix_status])
    gen_mode.change(
        fn=generate_mode_change,
        inputs=[gen_mode],
        outputs=[ui_topk, ui_verbose, gen_log, gen_verb, gen_underhood,
                 gen_branch_title, gen_branch_desc, branch_step_in, btn_branch,
                 branch_log, branch_html, branch_chart, btn_peek, peek_html],
    )
    btn_branch.click(fn=branch_from_step, inputs=[branch_step_in],
                     outputs=[branch_log, branch_html, branch_chart])
    btn_score.click(fn=score_text, inputs=[score_input],
                    outputs=[score_html, score_chart, score_stats])
    btn_compare.click(fn=compare_temperatures, inputs=[temp_prompt, temp_length],
                      outputs=[gallery_html])
    btn_replay.click(fn=run_replay, inputs=[replay_prompt],
                     outputs=[replay_log, replay_chart, replay_grid])
    replay_mode.change(fn=replay_mode_change, inputs=[replay_mode], outputs=[replay_log, replay_chart])
    replay_preset.change(
        fn=apply_replay_preset,
        inputs=[replay_preset],
        outputs=[replay_mode, replay_hint, replay_log, replay_chart],
    )
    btn_attn.click(fn=show_attention, inputs=[attn_ctx],
                   outputs=[attn_strip, attn_chart])
    audience_mode.change(
        fn=apply_audience_mode,
        inputs=[audience_mode],
        outputs=[audience_note, ui_verbose, wiz_advanced],
    )
    wiz_goal.change(
        fn=wizard_goal_change,
        inputs=[wiz_goal],
        outputs=[wiz_where, wiz_prompt, wiz_context, wiz_creativity, wiz_length, wiz_steps, wiz_lr],
    )
    wiz_start.click(
        fn=run_wizard,
        inputs=[wiz_goal, wiz_prompt, wiz_steps, wiz_lr, wiz_creativity, wiz_length, wiz_context, wiz_advanced, audience_mode],
        outputs=[wiz_status, wiz_where, wiz_main, wiz_extra, wiz_chart, wiz_next],
    )
    wiz_export.click(fn=export_latest_report, outputs=[wiz_export_status])
    hist_refresh.click(fn=refresh_history_view, outputs=[hist_timeline, hist_latest])
    hist_export.click(fn=export_latest_report, outputs=[hist_export_status])

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=".monospace textarea { font-family: monospace !important; font-size: 12px !important; }",
    )
