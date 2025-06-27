import torch
from tqdm import tqdm
import numpy as np
import constants
import model
import data_loader

# from helper import optimizer, scheduler, loss_fn


# train one epoch
def train_epoch(model, train_dataloader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0.0

    train_progress = tqdm(train_dataloader, desc="Training")
    for batch in train_progress:
        input_ids = batch["input_ids"].to(constants.DEVICE)
        attention_mask = batch["attention_mask"].to(constants.DEVICE)
        labels = batch["labels"].to(constants.DEVICE)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), constants.MAX_GRAD_NORM)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        train_progress.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_dataloader)
    print(f"Training Loss: {avg_loss}")
    return avg_loss


# evaluation fun
def evaluate(model, val_dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    val_progress = tqdm(val_dataloader, desc="Validation")

    with torch.no_grad():
        for batch in val_progress:
            input_ids = batch["input_ids"].to(constants.DEVICE)
            attention_mask = batch["attention_mask"].to(constants.DEVICE)
            labels = batch["labels"].to(constants.DEVICE)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            val_progress.set_postfix({"loss": loss.item()})

            prediction = torch.argmax(logits, dim=1)
            all_predictions.extend(prediction.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_dataloader)
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"Validation Loss: {avg_loss}")
    print(f"Validation Accuracy: {accuracy}")

    return avg_loss, accuracy
