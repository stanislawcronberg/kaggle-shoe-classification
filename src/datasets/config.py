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
    use_augmentations: bool
    use_early_stopping: bool
    early_stopping_kwargs: dict
    trainer_kwargs: dict
    dataloader_kwargs: dict


@dataclass
class Eval:
    ckpt_path: str  # Path to model checkpoint
    dataloader_kwargs: dict


@dataclass
class ShoeCLFConfig:
    data: Data
    model: str
    training: Training
    eval: Eval
    seed: int
