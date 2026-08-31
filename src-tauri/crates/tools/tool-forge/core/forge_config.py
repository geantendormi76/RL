import os
from typing import List
from pydantic import BaseModel, Field

class EvoHarnessConfig(BaseModel):
    task_families: List[str] = ["clean", "heat", "pick-two", "pick-place", "cool", "light"]
    
    sft_epochs: int = 3
    sft_batch_size: int = 1
    sft_grad_accum_steps: int = 8
    sft_learning_rate: float = 2e-5
    sft_max_seq_len: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    grpo_epochs: int = 150
    grpo_learning_rate: float = 1e-6
    grpo_group_size: int = 8
    kl_coeff: float = 0.01
    
    lambda_eff: float = 1.0
    lambda_div_max: float = 0.5
    lambda_spam: float = 0.1
    lambda_inv: float = 0.1

class ForgeConfig(BaseModel):
    project_root: str = r"C:\dev\RL"
    base_model_name_or_path: str = r"C:\dev\RL\models\Qwen3-8B"
    models_dir: str = r"C:\dev\RL\models"
    outputs_dir: str = r"C:\dev\RL\outputs"
    data_dir: str = r"C:\dev\RL\data"
    
    evoharness: EvoHarnessConfig = Field(default_factory=EvoHarnessConfig)

global_config = ForgeConfig()
