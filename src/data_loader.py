import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import constants


# load and prepare the dataset
def load_ag_news_dataset():
    try:
        # load dataset
        dataset = load_dataset(constants.DATASET_NAME)
        train_ds = dataset["train"]
        test_ds = dataset["test"]

        # convert to dataframe
        train_df = pd.DataFrame(train_ds)
        test_df = pd.DataFrame(test_ds)

        # Create id2label and label2id
        label_feature = dataset["train"].features["label"]
        label_names = label_feature.names
        id2label = {i: name for i, name in enumerate(label_names)}
        label2id = {name: i for i, name in enumerate(label_names)}

        # printing some infos
        print("Dataset loaded successfully")
        print(f"Training samples length: {len(train_df)}")
        print(f"Test samples length: {len(test_df)}")
        print(f"Dataset columns: {train_df.columns.tolist()}")
        print(f"Label mappings: {id2label}")
        print(f"First samples:\n{train_df.head(2)}")

        return train_df, test_df, id2label, label2id

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None, None


# create custom Dataset class
class ClfDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# prepare dataloader
def prepare_dataloader(train_df, tokenizer):
    texts = train_df[constants.TEXT_COLUMN].tolist()
    labels = train_df[constants.LABEL_COLUMN].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=constants.VALIDATION_SPLIT,
        random_state=constants.RANDOM_SEED,
    )

    train_dataset = ClfDataset(train_texts, train_labels, tokenizer)
    val_dataset = ClfDataset(val_texts, val_labels, tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=constants.BATCH_SIZE,
        shuffle=True,
        num_workers=constants.NUM_WORKERS,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=constants.BATCH_SIZE,
        shuffle=False,
        num_workers=constants.NUM_WORKERS,
    )

    return train_dataloader, val_dataloader


# test data loading functionality
def main():
    # Set random seed
    torch.manual_seed(constants.RANDOM_SEED)
    np.random.seed(constants.RANDOM_SEED)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME)
    # Load dataset
    train_df, test_df, id2label, label2id = load_ag_news_dataset()

    if train_df is not None:
        # Prepare data loaders
        train_dataloader, val_dataloader = prepare_dataloader(train_df, tokenizer)

        print(f"\nData loaders created successfully!")
        # Test a batchs of train and validation dataloder
        for batch in train_dataloader:
            print(f"Batch input_ids shape: {batch['input_ids'].shape}")
            print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
            print(f"Batch labels shape: {batch['labels'].shape}")
            break

        for batch in val_dataloader:
            print(f"Batch input_ids shape: {batch['input_ids'].shape}")
            print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
            print(f"Batch labels shape: {batch['labels'].shape}")
            break


if __name__ == "__main__":
    main()
