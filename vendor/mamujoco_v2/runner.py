"""Runner：把环境、算法、评估、日志和 checkpoint 串起来。

    while 步数 < total_env_steps:
        batch = collect()              # 并行环境各跑一条完整 episode
        metrics = algorithm.update(batch)
        按间隔：写 train 日志 / evaluate() / 存 checkpoint
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import random

import torch
import yaml

from utils.buffer import RolloutBatch

CHECKPOINT_FORMAT = "marl-mamujoco-v2"


class Runner:
    def __init__(self, config, algorithm, environment):
        self.config = config
        self.algorithm = algorithm
        self.environment = environment
        self.device = torch.device(config["device"])

    # ---------- 采样与评估 ----------

    @torch.no_grad()
    def collect(self, evaluation=False, reset_seed=None):
        """当前策略在 num_envs 个并行环境中各跑一条完整 episode。"""
        environment, algorithm = self.environment, self.algorithm
        spec = environment.spec
        tensor = lambda value, dtype=torch.float32: torch.as_tensor(value, device=self.device, dtype=dtype)
        observations_np, states_np = environment.reset(seed=reset_seed)
        observations, states = tensor(observations_np), tensor(states_np)
        batch_size = environment.batch_size
        hidden = algorithm.init_hidden(batch_size)
        shape = observations.new_zeros(batch_size, spec.n_agents, spec.n_actions)
        active = torch.ones(batch_size, 1, dtype=torch.bool, device=self.device)
        action_mask = algorithm.actor.action_mask.unsqueeze(0)
        settings = self.config["evaluation"]
        sample_latent = not evaluation or settings["sample_flow_noise"]     # latent 接口项（MAPPO / HAPPO / MAT 不使用）
        sample_noise = not evaluation or settings["sample_action_noise"]    # 终端高斯噪声 n；评估默认不加
        per_step = self.config.get("flow", {}).get("latent_resample", "step") == "step"

        rows = {name: [] for name in RolloutBatch.field_names()}
        rows["observations"].append(observations)
        rows["states"].append(states)
        latent = None
        for _ in range(spec.episode_limit):
            inputs = algorithm.actor_inputs(observations)
            if latent is None or per_step:
                latent = (torch.randn_like(shape) if sample_latent else torch.zeros_like(shape)) * action_mask
            sample, hidden = algorithm.act(inputs, hidden, latent, sample_noise)
            next_obs, next_states, rewards, terminated, truncated = environment.step(sample["actions"].cpu().numpy())
            terminated = tensor(terminated, torch.bool) & active
            truncated = tensor(truncated, torch.bool) & active
            rows["inputs"].append(inputs)
            for name in ("actions", "raw_actions", "means", "noise", "latent", "log_probs"):
                rows[name].append(sample[name])
            rows["rewards"].append(tensor(rewards) * active)
            rows["terminated"].append(terminated)
            rows["truncated"].append(truncated)
            rows["filled"].append(active.clone())
            observations, states = tensor(next_obs), tensor(next_states)
            rows["observations"].append(observations)
            rows["states"].append(states)
            active = active & ~(terminated | truncated)
            if not active.any():                    # 全部 episode 都已结束：后面只会是 padding，不再空跑到 episode_limit
                break
        if active.any():
            raise RuntimeError("环境在 episode_limit 后仍未 terminated/truncated")
        return RolloutBatch.stack(rows)

    @torch.no_grad()
    def evaluate(self):
        """固定环境 seed、独立随机数作用域，避免定期评估扰动训练。返回值为各 agent 奖励之和（team 回报 × N）。"""
        seed, episodes = self.config["seed"], self.config["evaluation"]["episodes"]
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        batches = []
        self.algorithm.train(False)
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed + 10000)
                while len(batches) * self.environment.batch_size < episodes:
                    done = len(batches) * self.environment.batch_size
                    batches.append(self.collect(evaluation=True, reset_seed=seed + 10000 + done))
        finally:
            self.algorithm.train(True)
        return {**episode_statistics(RolloutBatch.concatenate(batches, limit=episodes)), "episodes": episodes}

    # ---------- 训练主循环 ----------

    def run(self, resumed_from=None):
        config, algorithm = self.config, self.algorithm
        directory = self.make_run_directory(resumed_from)
        metrics_path = directory / "metrics.jsonl"
        print(f"algorithm={config['algorithm']} device={config['device']} "
              f"scenario={config['env']['scenario']}-{config['env']['agent_conf']} output={directory}", flush=True)

        log(metrics_path, "evaluate", algorithm.total_steps, self.evaluate())
        intervals = {
            "log": config["output"]["log_interval"],
            "evaluate": config["evaluation"]["interval"],
            "save": config["output"]["checkpoint_interval"],
        }
        due = {name: (algorithm.total_steps // every + 1) * every for name, every in intervals.items()}
        last_logged, last_evaluated, metrics = -1, algorithm.total_steps, None
        while algorithm.total_steps < config["training"]["total_env_steps"]:
            batches, collected = [], 0
            while not batches or collected < config["training"].get("batch_steps", 0):
                batches.append(self.collect(reset_seed=config["seed"] + algorithm.total_steps + 1 + collected))
                collected += batches[-1].env_steps
            batch = batches[0] if len(batches) == 1 else RolloutBatch.concatenate(batches)
            metrics = {**algorithm.update(batch), **episode_statistics(batch)}
            steps = algorithm.total_steps
            if steps >= due["log"]:
                log(metrics_path, "train", steps, metrics)
                last_logged = steps
            if steps >= due["evaluate"]:
                log(metrics_path, "evaluate", steps, self.evaluate())
                last_evaluated = steps
            if steps >= due["save"]:
                self.save(directory / f"step_{steps}.pt")
            for name, every in intervals.items():
                if steps >= due[name]:
                    due[name] = (steps // every + 1) * every

        if last_logged != algorithm.total_steps:
            log(metrics_path, "train", algorithm.total_steps, metrics)
        if last_evaluated != algorithm.total_steps:
            log(metrics_path, "evaluate", algorithm.total_steps, self.evaluate())
        self.save(directory / "latest.pt")
        print(f"完成。checkpoint: {directory / 'latest.pt'}", flush=True)

    def make_run_directory(self, resumed_from):
        """results/<算法>/<实验名>/seed_<seed>/<UTC 时间戳>/，写入本次的完整配置。"""
        config = self.config
        root = Path(config["output"]["directory"]).expanduser()
        if not root.is_absolute():
            root = Path(__file__).resolve().parent / root
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        directory = root / config["algorithm"] / config["output"]["experiment"] / f"seed_{config['seed']}" / timestamp
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if resumed_from is not None:
            (directory / "resume_from.txt").write_text(str(Path(resumed_from).resolve()) + "\n", encoding="utf-8")
        return directory

    # ---------- checkpoint ----------

    def save(self, path):
        payload = {
            "format": CHECKPOINT_FORMAT,
            "config": self.config,
            "algorithm_state": self.algorithm.checkpoint_state(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
        }
        temporary = Path(path).with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def restore(self, payload, resume):
        self.algorithm.load_checkpoint_state(payload["algorithm_state"], resume=resume)
        if resume:
            torch.set_rng_state(payload["torch_rng"].cpu())
            random.setstate(payload["python_rng"])
            if payload["cuda_rng"] is not None and self.device.type == "cuda":
                torch.cuda.set_rng_state_all([value.cpu() for value in payload["cuda_rng"]])


def episode_statistics(batch):
    """每条 episode 的回报（各 agent 奖励之和）与长度。"""
    returns = (batch.rewards * batch.filled).sum(dim=1).squeeze(-1)
    return {
        "return_mean": returns.mean().item(),
        "return_std": returns.std(unbiased=False).item(),
        "episode_length_mean": batch.filled.sum(dim=1).squeeze(-1).float().mean().item(),
    }


def log(path, kind, steps, metrics):
    record = {"kind": kind, "env_steps": steps, **metrics}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"[{kind:8s}] steps={steps:9d} return={metrics['return_mean']:.3f} "
          f"length={metrics['episode_length_mean']:.1f}", flush=True)
