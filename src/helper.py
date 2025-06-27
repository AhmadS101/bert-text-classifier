import os
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
import constants


# save model
def save_checkpoint(model, optimizer, epoch, loss, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filename)


# load model
def load_checkpoint(model, optimizer, filename):
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=constants.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Checkpoint loaded — resuming from epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found — starting from scratch.")
        return 0


# initiakize optimizer
def optimizer(model, criterion):
    optimizer = criterion(
        model.parameters(),
        lr=constants.LEARNING_RATE,
        weight_decay=constants.WEIGHT_DECAY,
    )

    return optimizer


# initiakize scheduler
def scheduler(train_dataloader):
    total_steps = len(train_dataloader) * constants.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(constants.WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    return scheduler


# initiakize loss function
def loss_fn(criterion):
    loss_fn = criterion

    return loss_fn
