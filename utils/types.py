import enum
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MOEBatch:
    input_ids: torch.Tensor
    initialization_token_id: int
    timesteps: torch.Tensor = None
    caption_ids: torch.Tensor = None
    task: str = None
    placeholder_token: Optional[str] = None
    image_features: torch.Tensor = None
    truncation_idx: Optional[int] = None    
    top_k: Optional[int] = None
    top_k_general: Optional[int] = None
    rescale: Optional[bool] = None
    layers : Optional[torch.Tensor] = None
    importance_loss_weight: Optional[float] = 0.0
    load_loss_weight: Optional[float] = 0.0
    sampled_norm : Optional[torch.Tensor]  = None
    global_step: Optional[int] = None
    


@dataclass
class PESigmas:
    sigma_t: float
    sigma_l: float
    