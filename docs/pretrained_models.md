# Pretrained Tello and M30 models

Eight reference checkpoints are included in the repository for Tello and M30 AI inference. They are the exact checkpoints used for both aircraft in the project report. Tello supports fast simulation and Gazebo playback; M30 supports fast simulation. Each checkpoint includes its original configuration and 100-episode evaluation results. Training used seed 1; evaluation used seeds 10,000–10,099.

| Aircraft | Algorithm | Wind | Fleet success | Individual success | Checkpoint |
|---|---|---:|---:|---:|---|
| TELLO | MAPPO | 0 m/s | 100% | 100.00% | [step_1500342.pt](../runs/marl/mappo/fixed_w0/seed_1/20261023T062020_041754Z/step_1500342.pt) |
| TELLO | PPO (IPPO) | 0 m/s | 99% | 99.75% | [latest.pt](../runs/marl/ippo/tello_w0_ippo_aligned/seed_1/20261025T065528_339108Z/latest.pt) |
| TELLO | MAPPO | 3 m/s | 90% | 97.50% | [latest.pt](../runs/marl/mappo/fixed_w3_ind/seed_1/20261023T070641_260637Z/latest.pt) |
| TELLO | PPO (IPPO) | 3 m/s | 91% | 97.75% | [latest.pt](../runs/marl/ippo/tello_w3_ippo_aligned/seed_1/20261025T065528_297224Z/latest.pt) |
| M30 | MAPPO | 0 m/s | 98% | 99.50% | [step_1500052.pt](../runs/marl/mappo/m30_xy5_z1_w0_ind/seed_1/20261024T102750_403975Z/step_1500052.pt) |
| M30 | PPO (IPPO) | 0 m/s | 98% | 99.50% | [step_1500629.pt](../runs/marl/ippo/m30_w0_ippo_aligned/seed_1/20261025T061033_329092Z/step_1500629.pt) |
| M30 | MAPPO | 10 m/s | 45% | 79.00% | [latest.pt](../runs/marl/mappo/m30_xy5_z1_w10_finetune/seed_1/20261024T122152_799106Z/latest.pt) |
| M30 | PPO (IPPO) | 10 m/s | 41% | 78.50% | [latest.pt](../runs/marl/ippo/m30_w10_ippo_aligned/seed_1/20261025T064900_015038Z/latest.pt) |

## Use in the GUI

Clone or pull the repository, install the dependencies described in the main README, and start:

```bash
python gui/web_server.py --port 18080
```

Select the aircraft, **AI Policy**, and the desired checkpoint. The GUI discovers these models automatically under `runs/marl/`. For **Tello**, choose **Gazebo** or **Fast simulation**; the wind-trained MAPPO model is `fixed_w3_ind`. For **M30**, choose **Fast simulation** and set mean wind to **10 m/s**; the wind-trained models are `m30_xy5_z1_w10_finetune` (MAPPO) and `m30_w10_ippo_aligned` (PPO/IPPO). Gazebo requires the system prerequisites described in the main README. The project currently has no M30 Gazebo model.

## Evaluate a checkpoint

From the repository root, after installing dependencies:

```bash
export PYTHONPATH="$PWD/src/tello_rl${PYTHONPATH:+:$PYTHONPATH}"
python -m tello_rl.evaluate_marl \
  runs/marl/mappo/fixed_w3_ind/seed_1/20261023T070641_260637Z/latest.pt \
  --episodes 100 --winds 3
```

The saved configuration is embedded in each `.pt` file; the accompanying `config.yaml` also enables GUI discovery. Original training output paths in those configurations are provenance and do not need to exist for inference. Model file sizes, SHA-256 checksums, and configuration/evaluation paths are listed in [the model manifest](pretrained_models.json).

To evaluate the M30 10 m/s MAPPO checkpoint, use:

```bash
python -m tello_rl.evaluate_marl \
  runs/marl/mappo/m30_xy5_z1_w10_finetune/seed_1/20261024T122152_799106Z/latest.pt \
  --episodes 100 --winds 10
```

The selected reference models are tracked explicitly; other training outputs remain ignored. The table reports each model at its listed wind speed; an evaluation file may also contain results for other wind speeds.
