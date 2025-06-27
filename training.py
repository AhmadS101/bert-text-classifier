import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import constants
import data_loader
import model as model_module
import trainer
import helper


# training loop
def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    epochs=constants.EPOCHS,
    start_epoch=0,
):
    torch.manual_seed(constants.RANDOM_SEED)
    np.random.seed(constants.RANDOM_SEED)

    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = trainer.train_epoch(
            model, train_dataloader, optimizer, scheduler, loss_fn
        )
        val_loss, val_accuracy = trainer.evaluate(model, val_dataloader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"Best validation accuracy: {best_val_accuracy}")

        helper.save_checkpoint(
            model, optimizer, epoch, val_loss, constants.CHECKPOINT_PATH
        )

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_accuracy": best_val_accuracy,
    }


# test training function
def main():
    # Set random seeds
    torch.manual_seed(constants.RANDOM_SEED)
    np.random.seed(constants.RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME)

    print("-" * 50)
    print("BERT Medical Classifier Training")
    print("=" * 40)
    print(f"Device: {constants.DEVICE}")
    print(f"Model: {constants.MODEL_NAME}")
    print(f"Batch size: {constants.BATCH_SIZE}")
    print(f"Learning rate: {constants.LEARNING_RATE}")
    print(f"Epochs: {constants.EPOCHS}")
    print("-" * 50)

    # load data
    print("Loading dataset...")
    train_df, test_df, id2label, label2id = data_loader.load_ag_news_dataset()

    print("Dataset loaded successfully")
    print(f"Training samples length: {len(train_df)}")
    print(f"Test samples length: {len(test_df)}")
    print(f"Dataset columns: {train_df.columns.tolist()}")
    print(f"Label mappings: {id2label}")
    print(f"First samples:\n{train_df.head(2)}")
    print("-" * 50)

    if train_df is None:
        print("Failed to load dataset. Exiting.")
        return

    print("Dataset loaded successfully")
    print(f"Training samples: {len(train_df)} | Test samples: {len(test_df)}")
    print(f"Labels: {id2label}")
    print("-" * 50)

    # prepare data loaders
    print("\nPreparing data loaders...")
    train_dataloader, val_dataloader = data_loader.prepare_dataloader(
        train_df, tokenizer
    )

    # create model
    print("\nInitializing model...")
    model = model_module.BERTClassifier().to(constants.DEVICE)

    # loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW
    total_steps = len(train_dataloader) * constants.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(constants.WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    # train model
    print("\nStarting training...")
    training_results = train(
        model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn
    )

    print("\nTraining completed successfully!")
    print(f"Final validation accuracy: {training_results['val_accuracies'][-1]:.4f}")


if __name__ == "__main__":
    main()
