# IPCAI submission — CAMMA team update

21 main slides for an informal 20–25-minute talk, followed by 7 vertically stacked technical backups. All slides have speaker notes. Updated 12 November 2026: expanded data heterogeneity and joint training sections. Paired phase images supplied by the presenter are included, pre-HCT above HCT. Point/line equations are schematic; exact scalar penalties and reductions need implementation verification.

## View locally

From the repository root:

```sh
python3 -m http.server 8000 --directory static
```

Open <http://localhost:8000/camma/meeting/2026.11.09/>. The deck uses the shared Reveal runtime in `static/camma/_reveal/`; serve the full `static` directory, not only this deck folder.

- Right / Space: advance (Space also traverses backups).
- Down / Up: navigate the backup stack.
- S: presenter view with notes and timer (allow the popup).
- F: fullscreen. O or Escape: overview.
- Discussion slide links jump directly to backups; each backup links back.
- Add `?print-pdf` for Reveal's browser print layout; fragments print together.

## Edit

- `index.html`: slide content, chart values, exact numerical tables, and speaker notes.
- `css/deck.css`: typography and layout.
- `js/deck.js`: Reveal configuration.
- `assets/`: source figures extracted losslessly from the supplied final paper, plus existing institutional logos.

No remote fonts, CDN scripts, new packages or build step are required. The deck is not published by creating these files.

## Sources and figures

Sources: `content/IPCAI26_main.pdf` and `content/IPCAI26_supplement.pdf`. Source references are kept in notes rather than on slides. Notes carry experimental details and caveats.

| Asset | Source |
|---|---|
| anatomy.png | Main Fig. 1, embedded image |
| ibsafe-key.png | Main Fig. 2a |
| ibsafe-uniform.png | Main Fig. 2b |
| ibsafe-mc.png | Main Fig. 2c |
| endoscapes.png | Main Fig. 2d |
| joint-prediction.png | Main Fig. 3a |
| weak-evidence.png | Main Fig. 3b |
| landmark-prediction.png | Main Fig. 3c |
| Institutional logos | Existing meeting/2026.04.30/assets/logo |

The supplement uses the older name “2C-ICG” for the source called “iBSAFE-MC” in the main paper; this deck consistently uses the main-paper name. Figures are retained as submitted, without new predictions or illustrative medical images. Charts use the reported values, on a 0–100% scale. The discussion slide contains proposed directions, not additional findings or commitments.

Phase strips: CMRP_4_preHCT.jpg and CMRP_4_HCT.jpg, moved from the project root. LoRA visual: supplied 01_sam3_lora.png, copied unchanged to assets/sam3-lora.png. Its box-stage configuration is distinguished from the final joint-model extension on the slide.
