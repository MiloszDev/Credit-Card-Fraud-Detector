"""Defines configuration models for the various data pipeline stages."""

from typing import List
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class GeneralConfig:
    seed: int
    device: str

@dataclass(frozen=True)
class DataConfig:
    raw_path: Path
    processed_path: Path
    test_split: float
    scaler: str

@dataclass(frozen=True)
class EnvironmentConfig:
    type: str
    steps: int

@dataclass(frozen=True)
class DqnAgentConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int
    gamma: float
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float
    learning_rate: float
    target_update_freq: int
    memory_size: int
    batch_size: int

@dataclass(frozen=True)
class TrainingConfig:
    num_episodes: int
    max_steps_per_episode: int
    log_interval: int
    save_model: bool
    model_save_path: Path

@dataclass(frozen=True)
class EvaluationConfig:
    metrics: List[str]
    threshold: float
    visualize: bool

@dataclass(frozen=True)
class LoggingConfig:
    use_wandb: bool
    wandb_project: str
    wandb_entity: str
    logger_name: str
    log_dir: Path