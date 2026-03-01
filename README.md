![BMDS Demo](output/visualizations/postprocessed_demo.gif)

# BMDS (Biomechanical Mouse Dynamics Synthesizer)

BMDS generates synthetic mouse trajectories that pass bot detection.

It covers Balabit telemetry ingestion, trajectory segmentation, kinematic feature extraction, MuJoCo physics simulation, offline RL policy training (IQL/CQL/BC), postprocessing, and quantitative evaluation against three independent bot detectors.

## Repository Layout

- `bmds/` — core package (data, environment, reward, training, utilities)
- `scripts/` — evaluation and visualization scripts (01–12)
- `run_training.py` — end-to-end pipeline entry point
- `data/` — raw and processed datasets (gitignored)
- `models/` — saved model files (gitignored)
- `output/` — visualizations and logs

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
npm install  # for DELBOT bot detector
```

## Quick Start

Full pipeline (download → dataset → train → evaluate):

```bash
python run_training.py
```

Recommended training (IQL, reuse existing dataset):

```bash
python run_training.py --skip-download --skip-dataset-build --algorithm iql --steps 100000
```

Generate trajectories from a trained model:

```bash
python scripts/06_generate_trajectories.py --start 100 100 --end 800 500 --plot
```

Run the full bot detection gauntlet (DELBOT RNN + GradBoost + One-Class SVM):

```bash
python scripts/11_multi_detector_gauntlet.py --n-movements 100 --seed 42
```

Visualize postprocessed output as an animated GIF:

```bash
python scripts/12_visualize_postprocessed.py --n 5
```

## Evaluation Scripts

| Script | Purpose |
|---|---|
| `09_bot_detection_test.py` | Feature-based GradBoost + OCSVM detector |
| `10_third_party_bot_test.py` | InceptionV3 image-based detector |
| `11_multi_detector_gauntlet.py` | All three detectors in one run |
| `12_visualize_postprocessed.py` | Animated GIF of postprocessed trajectories |

## Notes

- Primary algorithm: IQL (reaches 8/8 targets in ~2 min on RTX 4060)
- Postprocessing: oscillation truncation → Gaussian smoothing (σ=2.5, 3-frame boundary pin) → lateral arch → endpoint pauses with 0.28px tremor
- Main dependencies: MuJoCo 3.x, d3rlpy 1.1.x, PyTorch, Gymnasium, scikit-learn, Node.js (DELBOT)
