# Shakespeare GPT — Session Resume File

If you are a new Claude session picking this up, read this fully before touching any file.

---

## What exists and works (do not break these)

| File | Status | Purpose |
|---|---|---|
| `app.py` | ✅ Working | Main Gradio browser app — 3 tabs: Train, Generate, Score |
| `interact.py` | ✅ Working | Terminal REPL with verbose per-token logging |
| `tiny_shakespeare_gpt.py` | ✅ Working | Standalone training script |
| `tiny_shakespeare.pt` | ✅ Exists | Trained model weights (5000 steps) |
| `tinyshakespeare.txt` | ✅ Exists | 1.1M char Shakespeare dataset |
| `userguide.html` | ✅ Updated | Full user guide reflecting all current features |
| `launch.sh` | ✅ Working | One-click launcher (creates venv, installs deps) |
| `.venv/` | ✅ Working | Python env with torch, gradio, tensorboard, matplotlib |

## Current app.py features (already built)

1. **Train tab** — train or load model, live loss chart
2. **Generate tab** — verbose token log + confidence-coloured text + entropy chart + branch
3. **Score tab** — paste text, see per-character confidence + perplexity
4. Human-friendly language throughout (no jargon)
5. Verbatim detection (n-gram check) — built in global state, ready to use

## What still needs to be built (the task that was interrupted)

Four new features for a product-owner / non-technical AI researcher persona.
Build them in this order:

### 1. 🌡️ Temperature Gallery (PRIORITY 1 — easiest, highest impact)
**New tab in app.py**
- User types a prompt + length slider (20–150)
- Click "Compare" → generates same prompt at 5 temperatures: 0.3 / 0.6 / 1.0 / 1.4 / 1.8
- Shows a 5-card grid, each card showing confidence-coloured text + avg confidence badge
- Cards appear one by one as each temperature finishes (streaming)
- Key function: `compare_temperatures(prompt, length)` → yields HTML
- Key helper: `_build_gallery_html(prompt, results)` → returns HTML string
- UX copy: temperature labels should be human ("Focused", "Careful", "Balanced", "Creative", "Wild")

### 2. ⏪ Training Replay (PRIORITY 2)
**New tab in app.py + modify stream_training to save checkpoints**

Two parts:
**Part A**: Modify `stream_training()` to save checkpoints to `checkpoints/` dir at steps:
100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000 (only if step ≤ n_steps)

**Part B**: `run_replay(test_phrase, gen_prompt)` generator function:
- Scans `checkpoints/` for saved `.pt` files
- If none found: run a quick 1000-step training with checkpoints at every 100 steps first
- For each checkpoint: load model, compute perplexity on test_phrase, generate 80 chars
- Yield after each checkpoint: progress text + perplexity chart + sample grid HTML
- Perplexity chart: matplotlib, x=training step, y=perplexity, dark theme
- Sample grid: HTML cards (one per checkpoint), each showing the generated text coloured by confidence

Checkpoint format: `checkpoints/step_00100.pt` (zero-padded)

### 3. 🔍 Verbatim Detector (PRIORITY 3 — integrate into Generate tab)
**Add as extra output panel in Generate tab, no new tab needed**

At startup, build a set of all 6-grams from training text:
```python
_ngrams = set(_text[i:i+6] for i in range(len(_text) - 6))
```

After generation in `stream_generate`, compute verbatim mask:
- For each position i: True if generated_text[i:i+6] is in _ngrams
- Show as a second coloured HTML: blue/purple = in training data, normal = novel
- Add a summary line: "X% of generated text appears in the training data"
- Add `verbatim_html` output component in Generate tab UI

Key functions to add:
- `compute_verbatim_mask(text, ngrams)` → list of bools
- `make_verbatim_html(prompt, generated, mask)` → HTML string

### 4. 🔭 Attention Heatmap (PRIORITY 4 — requires model modification)
**New tab + modify CausalSelfAttention**

**Model modification**: Add to CausalSelfAttention:
```python
self._capture = False
self._last_attn = None  # stores (n_head, T, T) when _capture=True
```
In forward(), after softmax but before dropout:
```python
if self._capture:
    self._last_attn = att[0].detach().cpu()
```

**Helper**: `capture_attention(model, context_str)`:
- Encode context_str to tokens
- Enable capture on all blocks
- Run forward pass
- Disable capture
- Return list of (n_head, T, T) tensors (one per layer)

**Visualization** (human-friendly, not academic):
- Top section: context characters as a horizontal strip, each coloured by avg attention weight it received (bright = model focused here)
- Bottom: 4-panel heatmap grid, one per layer, with char labels on x-axis
- Natural language summary: "When deciding the next character, the model focused most on 'X' (NN%) and 'Y' (NN%)"
- Use last 18 chars of context max (otherwise too wide to display)

Key functions:
- `show_attention(context_str)` → (html_strip, chart_4panel)
- `make_attention_strip_html(chars, avg_attn_per_char)` → HTML
- `make_attention_chart(chars, attn_layers)` → matplotlib Figure (2×2 subplots)

---

## Implementation notes

- Run: `.venv/bin/python app.py`  (Gradio 6.8.0 on Python 3.14)
- Gradio 6 quirk: `theme` and `css` go in `launch()` not `Blocks()`
- Matplotlib quirk: colours must be hex `#rrggbb` not `rgb(r,g,b)` strings
- All new tabs follow the same dark theme: bg `#0f1117`, card `#1a1d27`
- The n-gram set build happens at module level (at import time) — acceptable ~1-2 sec startup
- `_gen_history` global stores idx tensors for branching — persists across generate calls
- Test with: `.venv/bin/python test_output.py` (create minimal test file, delete after)

## User context
- Product owner, non-technical AI researcher, 36 years old
- Wants to UNDERSTAND how LLMs work, not build them
- All UI text must be in plain English — no jargon
- Business analogies work very well for this user
- The app runs locally on their Mac (darwin, Apple Silicon)

## To resume this session
1. Read this file
2. Read the current `app.py` (540 lines) to understand structure
3. Build features in priority order: Temperature → Replay → Verbatim → Attention
4. Keep all existing features working (do not change Train/Generate/Score tabs)
5. Update `userguide.html` after all features are built
6. Update `launch.sh` if any new packages are needed (none expected)
