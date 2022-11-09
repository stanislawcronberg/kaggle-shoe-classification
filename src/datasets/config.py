from dataclasses import dataclass


@dataclass
class Data:
    index: dict[str, str]
    image_size: tuple[int, int]
    n_classes: int
    in_channels: int


@dataclass
class Training:
    learning_rate: float
    batch_size: int
    use_augmentations: bool
    use_early_stopping: bool
    early_stopping_kwargs: dict
    trainer_kwargs: dict
    num_workers: int
    pin_memory: bool


@dataclass
class Eval:
    batch_size: int
    ckpt_path: str  # Path to model checkpoint


@dataclass
class ShoeCLFConfig:
    data: Data
    model: str
    training: Training
    seed: int
