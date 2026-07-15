# Ludus asset packs

Every file enters through its pack's `manifest.json` (see docs/10). No
manifest entry -> `scripts/asset_audit.py` fails CI. AI-generated entries
record tool/model/prompt/date as provenance.

Drop-in flow for owner-generated images (GPT/Gemini/etc):
1. Save as `assets/<pack>/<key>.png` using the exact key from the prompt pack.
2. Add the manifest entry (template below) — or run
   `python scripts/asset_intake.py <pack> <file> --tool gpt4o --prompt-id T1`.
3. `python scripts/asset_audit.py` must pass; the renderer picks the file up
   by key with no code change (placeholder fallback disappears).
