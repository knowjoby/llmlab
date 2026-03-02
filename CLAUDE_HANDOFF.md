# Claude Code Handoff — Wizard Update

Date: March 2, 2026 (Monday)  
Planned resume date: March 6, 2026 (Friday)

## Context
This handoff records the Wizard-focused UX changes made in `app.py` for the Tiny Shakespeare GPT Gradio app.

## What was changed
- Added a new guided `🧭 Wizard` tab to help non-coding users follow a clear flow:
  - Step 1: Choose goal
  - Step 2: Enter only required inputs
  - Step 3: Run
  - Step 4: View outputs
  - Step 5: See suggested next actions
- Implemented Wizard helper logic:
  - `wizard_step_guide(goal)`
  - `wizard_goal_change(goal)` (dynamic show/hide of inputs by goal)
  - `wizard_next_actions(goal)`
  - `run_wizard(...)` (routes to train / generate / compare / attention flows)
- Added a simple creativity mapping for PO-friendly control:
  - `_wizard_creativity_to_temp(creativity)` maps `0..100` to temperature `0.3..1.8`
- Added model readiness helper:
  - `_wizard_load_model_if_needed()`

## Files touched
- `/Users/jobyjohn/llm/app.py`

## Event wiring added
- Wizard goal change event now updates:
  - guidance text
  - visibility of prompt/context/creativity/length/training fields
- Wizard start event now outputs:
  - status
  - guidance
  - main output
  - extra details
  - chart/visual
  - next-step recommendations

## Validation performed
- Syntax check passed:
  - `.venv/bin/python -m py_compile app.py`
- Import and function smoke checks passed with Gradio launch patched:
  - verified `run_wizard`, `wizard_goal_change`, `wizard_next_actions`
  - verified wizard yields expected output tuple size
  - verified Generate-path wizard run returns HTML + chart object

## Notes for Friday resume
- Primary next improvement: add `Back` / `Reset Wizard` controls for cleaner step navigation.
- Optional UX polish:
  - dynamically disable irrelevant controls (in addition to hiding)
  - add short helper text under each visible field
  - add one-click “Go to classic tab” links after each wizard run

## Change log (latest)
- Added critical product features:
  - Run history timeline tab (`📚 History`)
  - Quality scorecard + guardrail tips after generation
  - Audience mode toggle (`Executive` / `Research`)
  - Report export (`runs/reports/*.md`)
- Added persistent history storage:
  - `runs/run_history.json` is auto-saved and auto-loaded
- Added “under the hood” explainability:
  - Plain-English generation breakdown panel
  - Next-character peek (top-5 predicted characters)
- Simplified Generate UI:
  - New `Simple` vs `Advanced` view mode
  - Simple mode hides verbose/branch/peek controls
  - Advanced mode reveals full controls for deeper exploration
- Extended simplified UI pattern:
  - Train tab now has `Simple` vs `Advanced` mode (learning-rate + train log hidden in Simple)
  - Replay tab now has `Simple` vs `Advanced` mode (progress + perplexity chart hidden in Simple)
- Documentation sync:
  - `userguide.html` updated with Train/Replay view-mode notes
  - `userguide.html` "What’s new" section includes Train/Replay simplification
- Added pragmatic iteration helpers in Generate:
  - `Compare with last run` card in under-the-hood panel
  - `Apply recommended fix` button to auto-adjust settings from guardrail recommendation
