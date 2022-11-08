from dataclasses import dataclass


@dataclass
class Data:
    data_dir: str
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


@dataclass
class ShoeCLFConfig:
    data: Data
    training: Training
    seed: int
