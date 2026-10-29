# Pretrained Tello models

Four reference checkpoints are included in the repository for Tello AI inference and Gazebo playback. They are the exact checkpoints used for the Tello results in the project report. Each checkpoint includes its original configuration and 100-episode evaluation results. Training used seed 1; evaluation used seeds 10,000–10,099.

| Algorithm | Wind | Fleet success | Individual success | Checkpoint |
|---|---:|---:|---:|---|
| MAPPO | 0 m/s | 100% | 100.00% | [step_1500342.pt](../runs/marl/mappo/fixed_w0/seed_1/20261023T062020_041754Z/step_1500342.pt) |
| PPO (IPPO) | 0 m/s | 99% | 99.75% | [latest.pt](../runs/marl/ippo/tello_w0_ippo_aligned/seed_1/20261025T065528_339108Z/latest.pt) |
| MAPPO | 3 m/s | 90% | 97.50% | [latest.pt](../runs/marl/mappo/fixed_w3_ind/seed_1/20261023T070641_260637Z/latest.pt) |
| PPO (IPPO) | 3 m/s | 91% | 97.75% | [latest.pt](../runs/marl/ippo/tello_w3_ippo_aligned/seed_1/20261025T065528_297224Z/latest.pt) |

## Use in the GUI

Clone or pull the repository, install the dependencies described in the main README, and start:

```bash
python gui/web_server.py --port 18080
```

Select **Tello**, **AI Policy**, the desired checkpoint, and **Gazebo**. The GUI discovers these models automatically under `runs/marl/`. Gazebo requires the system prerequisites described in the main README. Fast simulation can use the same checkpoints without Gazebo. For the existing wind-trained GUI default, choose `fixed_w3_ind`.

## Evaluate a checkpoint

From the repository root, after installing dependencies:

```bash
export PYTHONPATH="$PWD/src/tello_rl${PYTHONPATH:+:$PYTHONPATH}"
python -m tello_rl.evaluate_marl \
  runs/marl/mappo/fixed_w3_ind/seed_1/20261023T070641_260637Z/latest.pt \
  --episodes 100 --winds 3
```

The saved configuration is embedded in each `.pt` file; the accompanying `config.yaml` also enables GUI discovery. Original training output paths in those configurations are provenance and do not need to exist for inference. Model file sizes, SHA-256 checksums, and configuration/evaluation paths are listed in [the model manifest](pretrained_models.json).

The selected reference models are tracked explicitly; other training outputs remain ignored. M30 models are not included in this Tello bundle.
