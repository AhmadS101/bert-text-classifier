import os
import torch
import constants
import matplotlib.pyplot as plt
import seaborn as sns


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


# print model preformance with test set
def print_metrics(clf_report, conf_mat, id2label):
    print(f"\nModel Accuracy: {clf_report['accuracy']:.2%}\n")

    # Print per-class metrics
    print(
        "{:<15} {:<10} {:<10} {:<10}".format("Class", "Precision", "Recall", "F1-Score")
    )
    print("-" * 45)

    for i, label in id2label.items():
        print(
            "{:<15} {:<10.2%} {:<10.2%} {:<10.2%}".format(
                label,
                clf_report[label]["precision"],
                clf_report[label]["recall"],
                clf_report[label]["f1-score"],
            )
        )

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(conf_mat)


# print model confusion matrix
def plot_confusion_matrix(conf_mat, id2label):
    """Visualize confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        xticklabels=id2label.values(),
        yticklabels=id2label.values(),
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
