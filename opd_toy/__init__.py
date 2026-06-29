"""Offline on-policy distillation under multi-turn agentic distribution shift."""

from .env import EnvConfig, RetrievalQAEnv
from .policies import LinearSoftmaxStudent, TabularStudent, TeacherPolicy, build_features
from .methods import (
    TrainConfig,
    train_sft,
    train_offline_opd,
    train_online_opd,
    train_online_rl,
    train_refresh,
    theorem_diagnostics,
)

__all__ = [
    "EnvConfig",
    "RetrievalQAEnv",
    "TeacherPolicy",
    "LinearSoftmaxStudent",
    "TabularStudent",
    "build_features",
    "TrainConfig",
    "train_sft",
    "train_offline_opd",
    "train_online_opd",
    "train_online_rl",
    "train_refresh",
    "theorem_diagnostics",
]
