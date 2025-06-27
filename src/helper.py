import os
import torch
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
