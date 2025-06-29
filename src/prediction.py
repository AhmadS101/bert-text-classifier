import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import constants
import data_loader
import model
import helper
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix


def model_evaluation(
    model, test_dataloader, loss_fn, id2label, device=constants.DEVICE
):
    model.eval()
    total_loss = 0.0
    predictions = []
    t_labels = []
    test_progress = tqdm(test_dataloader, desc="Validation")

    with torch.no_grad():
        for batch in test_progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            test_progress.set_postfix({"loss": loss.item()})

            _, pred = torch.max(outputs, dim=1)

            predictions.extend(pred.cpu().tolist())
            t_labels.extend(labels.cpu().tolist())

    class_names = [id2label[i] for i in sorted(id2label.keys())]
    clf_report = classification_report(
        t_labels, predictions, target_names=class_names, digits=4, output_dict=True
    )

    conf_mat = confusion_matrix(t_labels, predictions)

    avg_loss = total_loss / len(test_dataloader)
    accuracy = np.mean(np.array(predictions) == np.array(t_labels))
    print(f"Test-Set Loss: {avg_loss}")
    print(f"Test-Set Accuracy: {accuracy}")

    return clf_report, conf_mat, predictions, t_labels


def main():
    # Load your trained model and tokenizer
    model = model.BERTClassifier().to(constants.DEVICE)
    model.load_state_dict(torch.load("../model/bert_clf_model.ptrom"))
    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME)

    # Load test data
    _, test_df, id2label, label2id = data_loader.load_ag_news_dataset()

    # Create test dataloader
    test_texts = test_df[constants.TEXT_COLUMN].tolist()
    test_labels = test_df[constants.LABEL_COLUMN].tolist()
    test_dataset = data_loader.ClfDataset(test_texts, test_labels, tokenizer)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=constants.BATCH_SIZE,
        shuffle=True,
        num_workers=constants.NUM_WORKERS,
    )

    # loss function
    loss_fn = nn.CrossEntropyLoss()
    # Evaluate model
    clf_report, conf_mat, predictions, t_labels = model_evaluation(
        model, test_dataloader, loss_fn, id2label, device=constants.DEVICE
    )

    # Print and visualize results
    helper.print_metrics(clf_report, conf_mat, id2label)
    helper.plot_confusion_matrix(conf_mat, id2label)

    # Additional statistics
    print(f"\nTotal Test Samples: {len(t_labels)}")
    print(f"Class Distribution: {np.bincount(t_labels)}")


if __name__ == "__main__":
    main()
