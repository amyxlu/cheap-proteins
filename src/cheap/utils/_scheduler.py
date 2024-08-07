from transformers import (
    get_scheduler,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    Adafactor,
)
import torch


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_type: str = "constant",
    num_warmup_steps: int = 0,
    num_training_steps: int = 10_000_000,
    num_cycles: int = 1,
):
    # Set this to something where the scaling factor is actually meaningful
    if sched_type == "cosine_with_restarts":
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )
    elif sched_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )
    else:
        return get_scheduler(
            name=sched_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
