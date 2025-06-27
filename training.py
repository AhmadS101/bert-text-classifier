import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
import constants
import data_loader
import model
import trainer
import helper

criterion = nn.CrossEntropyLoss()
train_dataloader, val_dataloader = data_loader.prepare_dataloader()
model = model.BERTClassifier()
loss_fn = helper.loss_fn(criterion)
optimizer = helper.optimizer(model, criterion)
save_checkpoint = helper.save_checkpoint()


# training loop
def train(train_dataloader, val_dataloader, epochs=constants.EPOCHS, start_epoch=0):
    torch.manual_seed(constants.RANDOM_SEED)
    np.random.seed(constants.RANDOM_SEED)

    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0

    for epoch in range(start_epoch, epochs):
        train_loss = trainer.train_epoch(train_dataloader)
        val_loss, val_accuracy = trainer.evaluate(val_dataloader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"Best validation accuracy: {best_val_accuracy}")

        save_checkpoint(model, optimizer, epoch, val_loss, constants.CHECKPOINT_PATH)
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

    # prepare data loaders
    print("\nPreparing data loaders...")
    train_dataloader, val_dataloader = data_loader.prepare_dataloader(
        train_df, tokenizer
    )

    # create model
    print("\nInitializing model...")
    model = model.BERTClassifier()

    # train model
    print("\nStarting training...")
    training_results = train()

    print("\nTraining completed successfully!")
    print(f"Final validation accuracy: {training_results['val_accuracies'][-1]:.4f}")


if __name__ == "__main__":
    main()
