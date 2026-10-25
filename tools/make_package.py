"""Make package."""
import argparse
import shutil
import tarfile
from datetime import date
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
MARL_CORE = WS / 'vendor' / 'mamujoco_v2'
MARL_FILES = ['runner.py', 'algorithms/__init__.py', 'algorithms/mappo.py', 'algorithms/ippo.py', 'algorithms/happo.py', 'algorithms/mat.py',
              'models/__init__.py', 'models/mappo.py', 'models/happo.py', 'models/mat.py', 'models/common_critic.py',
              'utils/__init__.py', 'utils/buffer.py', 'utils/normalization.py', 'configs/config.yaml']
IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path, help='MAPPO / HAPPO checkpoint(latest.pt)')
    parser.add_argument('--videos', nargs='*', type=Path, default=[], help='Demo video to include (mp4 or gif)')
    parser.add_argument('--out', type=Path, default=WS / 'dist')
    args = parser.parse_args()

    name = f'tello_multi_landing_{date.today():%Y%m%d}'
    stage = args.out / name
    if stage.exists():
        raise SystemExit(f'{stage} already exists; move it or select another output directory')
    stage.mkdir(parents=True)

    shutil.copytree(WS / 'src', stage / 'src', ignore=IGNORE)
    for rel in MARL_FILES:
        target = stage / 'vendor' / 'mamujoco_v2' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MARL_CORE / rel, target)

    (stage / 'models').mkdir()
    shutil.copy2(args.model, stage / 'models' / 'mappo_seq.pt')
    (stage / 'results').mkdir()
    for path in sorted(args.model.parent.glob('eval_*.json')) + [args.model.parent / 'metrics.jsonl']:
        if path.exists():
            shutil.copy2(path, stage / 'results' / path.name)
    for video in args.videos:
        shutil.copy2(video, stage / 'results' / video.name)
    for doc in ('README.md', 'requirements.txt'):
        shutil.copy2(WS / doc, stage / doc)

    archive = args.out / f'{name}.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(stage, arcname=name)
    size = archive.stat().st_size / 1e6
    print(f'-> {archive}  ({size:.1f} MB)')


if __name__ == '__main__':
    main()
