"""
Interactive inference for Tiny Shakespeare GPT.
Shows what the model is thinking at every token step.

Usage:
    python interact.py [--temp 1.0] [--length 200] [--topk 5] [--quiet]

Commands inside the REPL:
    temp=0.8     change temperature
    len=150      change generation length
    topk=3       change how many candidates to show
    verbose      toggle per-token logging (default: on)
    quit / exit  quit
"""

import os
import math
import argparse
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

# ── Config (must match training) ─────────────────────────────────────────────
BLOCK_SIZE = 128
N_EMBD     = 64
N_HEAD     = 4
N_LAYER    = 4
DROPOUT    = 0.0          # inference: no dropout
MODEL_PATH = "tiny_shakespeare.pt"
DATA_PATH  = "tinyshakespeare.txt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Rebuild vocab from dataset ────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found — run tiny_shakespeare_gpt.py first.")

with open(DATA_PATH) as f:
    text = f.read()

chars     = sorted(set(text))
vocab_size = len(chars)
stoi      = {c: i for i, c in enumerate(chars)}
itos      = {i: c for i, c in enumerate(chars)}
encode    = lambda s: [stoi[c] for c in s if c in stoi]
decode    = lambda l: "".join(itos[i] for i in l)

# ── Model (identical architecture to training script) ────────────────────────
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
               .view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(N_EMBD, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y   = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
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
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.head    = nn.Linear(N_EMBD, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=DEVICE)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x    = self.blocks(x)
        x    = self.ln_f(x)
        return self.head(x)        # (B, T, vocab_size)


# ── Load weights ──────────────────────────────────────────────────────────────
model = GPT().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f"Loaded {MODEL_PATH}  ({sum(p.numel() for p in model.parameters()):,} params)")
print(f"Vocab: {vocab_size} chars  |  Device: {DEVICE}\n")

# ── Helper: entropy in bits ───────────────────────────────────────────────────
def entropy_bits(probs: torch.Tensor) -> float:
    p = probs.clamp(min=1e-10)
    return (-( p * p.log2() ).sum()).item()

# ── Core: verbose generate ────────────────────────────────────────────────────
def generate_verbose(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k_display: int,
    verbose: bool,
    writer: SummaryWriter,
    session_id: int,
):
    tokens = encode(prompt)
    if not tokens:
        tokens = [0]

    idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    # per-step stats collected for summary + TensorBoard
    entropies   = []
    confidences = []
    chosen_ranks = []

    if verbose:
        print()
        print("─" * 68)
        print(f"  Prompt  : {repr(prompt)}")
        print(f"  Temp    : {temperature}  |  Generating {max_new_tokens} tokens")
        print("─" * 68)

    generated_tokens = []

    with torch.no_grad():
        for step in range(1, max_new_tokens + 1):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits   = model(idx_cond)[:, -1, :]          # (1, vocab)

            # Apply temperature
            scaled_logits = logits / max(temperature, 1e-6)
            probs          = F.softmax(scaled_logits, dim=-1).squeeze(0)  # (vocab,)

            # Entropy & top-k
            H = entropy_bits(probs)

            top_probs, top_ids = probs.topk(top_k_display)
            top_probs = top_probs.cpu().tolist()
            top_ids   = top_ids.cpu().tolist()

            # Sample
            chosen_id    = torch.multinomial(probs, num_samples=1).item()
            chosen_prob  = probs[chosen_id].item()
            chosen_char  = itos[chosen_id]

            # Rank of chosen token among all (1 = argmax)
            rank = int((probs > probs[chosen_id]).sum().item()) + 1

            entropies.append(H)
            confidences.append(chosen_prob)
            chosen_ranks.append(rank)

            # ── TensorBoard per-step ──────────────────────────────────────
            tb_step = (session_id - 1) * max_new_tokens + step
            writer.add_scalar("inference/entropy_bits",  H,           tb_step)
            writer.add_scalar("inference/confidence",    chosen_prob, tb_step)
            writer.add_scalar("inference/chosen_rank",   rank,        tb_step)
            writer.add_scalar("inference/top1_prob",     top_probs[0], tb_step)

            # ── Console per-token ─────────────────────────────────────────
            if verbose:
                ctx_so_far = decode(idx[0].tolist()[-20:])   # last 20 chars for context
                print(f"\n  Step {step:>3} | Context tail: {repr(ctx_so_far)}")
                print(f"           Top {top_k_display} candidates:")
                for rank_i, (pid, pp) in enumerate(zip(top_ids, top_probs), 1):
                    marker = " ← chosen" if pid == chosen_id else ""
                    bar    = "█" * int(pp * 40)
                    print(f"             {rank_i}. {repr(itos[pid]):>6}  "
                          f"prob={pp:.4f}  {bar}{marker}")
                if chosen_id not in top_ids:
                    print(f"             *  {repr(chosen_char):>6}  "
                          f"prob={chosen_prob:.4f}  ← chosen (rank {rank})")
                print(f"           Entropy: {H:.3f} bits  |  "
                      f"Confidence: {chosen_prob*100:.1f}%  |  Rank: {rank}")

            # Append chosen token
            idx = torch.cat([idx, torch.tensor([[chosen_id]], device=DEVICE)], dim=1)
            generated_tokens.append(chosen_id)

    generated_text = decode(generated_tokens)

    # ── Summary stats ─────────────────────────────────────────────────────────
    avg_H    = sum(entropies)   / len(entropies)
    avg_conf = sum(confidences) / len(confidences)
    max_H_step = entropies.index(max(entropies)) + 1
    min_H_step = entropies.index(min(entropies)) + 1

    writer.add_scalar("inference/session_avg_entropy",    avg_H,    session_id)
    writer.add_scalar("inference/session_avg_confidence", avg_conf, session_id)
    writer.add_text(
        "inference/output",
        f"**Prompt:** {repr(prompt)}\n\n**Generated:**\n```\n{generated_text}\n```",
        session_id,
    )

    print()
    print("━" * 68)
    print(f"  OUTPUT")
    print("━" * 68)
    print(prompt + generated_text)
    print()
    print("━" * 68)
    print(f"  GENERATION SUMMARY (session {session_id})")
    print("━" * 68)
    print(f"  Tokens generated  : {max_new_tokens}")
    print(f"  Temperature       : {temperature}")
    print(f"  Avg entropy       : {avg_H:.3f} bits  "
          f"(max={max(entropies):.3f} @ step {max_H_step}, "
          f"min={min(entropies):.3f} @ step {min_H_step})")
    print(f"  Avg confidence    : {avg_conf*100:.1f}%")
    print(f"  Rank-1 chosen     : "
          f"{sum(1 for r in chosen_ranks if r==1)}/{max_new_tokens} tokens  "
          f"({100*sum(1 for r in chosen_ranks if r==1)/max_new_tokens:.0f}%)")
    print(f"  TensorBoard       : session {session_id} logged to runs/inference")
    print("━" * 68)

    return generated_text

# ── REPL ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Interact with Tiny Shakespeare GPT")
    p.add_argument("--temp",   type=float, default=1.0,  help="Sampling temperature")
    p.add_argument("--length", type=int,   default=200,  help="Tokens to generate")
    p.add_argument("--topk",   type=int,   default=5,    help="Top-k candidates shown")
    p.add_argument("--quiet",  action="store_true",      help="Hide per-token logs")
    return p.parse_args()

def main():
    args    = parse_args()
    temp    = args.temp
    length  = args.length
    topk    = args.topk
    verbose = not args.quiet

    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer  = SummaryWriter(log_dir=f"runs/inference/{run_tag}")
    session = 0

    print("┌─────────────────────────────────────────────────────────────┐")
    print("│            Tiny Shakespeare GPT  ·  Interactive             │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  Type a prompt and press Enter to generate.                 │")
    print("│  Settings:  temp=<n>  len=<n>  topk=<n>  verbose           │")
    print("│  Quit:      quit | exit | Ctrl-C                            │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"  Current settings → temp={temp}  len={length}  topk={topk}  "
          f"verbose={'on' if verbose else 'off'}")
    print()

    while True:
        try:
            raw = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit"):
            print("Bye.")
            break

        # ── Setting adjustments ───────────────────────────────────────────
        changed = False
        if raw.startswith("temp="):
            try:
                temp = float(raw.split("=", 1)[1])
                print(f"  → temperature = {temp}")
                changed = True
            except ValueError:
                print("  Bad value for temp.")
        elif raw.startswith("len="):
            try:
                length = int(raw.split("=", 1)[1])
                print(f"  → length = {length}")
                changed = True
            except ValueError:
                print("  Bad value for len.")
        elif raw.startswith("topk="):
            try:
                topk = int(raw.split("=", 1)[1])
                print(f"  → topk = {topk}")
                changed = True
            except ValueError:
                print("  Bad value for topk.")
        elif raw.lower() == "verbose":
            verbose = not verbose
            print(f"  → verbose {'on' if verbose else 'off'}")
            changed = True

        if changed:
            continue

        # ── Generate ──────────────────────────────────────────────────────
        session += 1
        generate_verbose(
            prompt=raw,
            max_new_tokens=length,
            temperature=temp,
            top_k_display=topk,
            verbose=verbose,
            writer=writer,
            session_id=session,
        )

    writer.close()
    print(f"\nAll sessions logged → tensorboard --logdir=runs/inference/{run_tag}")

if __name__ == "__main__":
    main()
