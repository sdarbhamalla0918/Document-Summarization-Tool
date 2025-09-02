from dataclasses import dataclass
import yaml

@dataclass
class TrainConfig:
    model_name_or_path: str
    output_dir: str
    max_source_length: int = 512
    max_target_length: int = 128
    min_target_length: int = 32
    num_train_epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    logging_steps: int = 50
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 1000
    fp16: bool = True
    seed: int = 42

    @staticmethod
    def from_yaml(path: str):
        with open(path) as f:
            data = yaml.safe_load(f)
        return TrainConfig(**data)
