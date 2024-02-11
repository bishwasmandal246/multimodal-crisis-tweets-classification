import os
import torch
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from config import plots_path


def evaluate_finetuning(model,
                        task,
                        finetuned_model_name,
                        logger,
                        test_dataloader):
    '''
    Evaluate finetuning results of CrisisMMD dataset.

    Arguments
    ---------
    model: model to be evaluated.
    task: (str) Humanitarian or Informative.
    finetuned_model_name: (str) name of the finetuned model that is evaluated.
    save_logs_filename: (str) file where you want to store the results.
    test_dataloader: test data on which results are to be evaluated.
    '''

    actual, predicted = [], []
    model.eval()
    with torch.no_grad():
        for batch, labels in test_dataloader:
            labels = labels.cpu()
            outputs = model(**batch).cpu()
            outputs = np.argmax(outputs, axis=1)
            actual.extend(labels)
            predicted.extend(outputs)

    if logger:
        logger.info(f"Classification Results, Task: {task}, Finetuned Model: {finetuned_model_name}")

    logger.info(f"Accuracy: {accuracy_score(actual, predicted)*100:.2f}")
    logger.info(f"Precision: {precision_score(actual, predicted, average='weighted')*100:.2f}")
    logger.info(f"Recall: {recall_score(actual, predicted, average='weighted')*100:.2f}")
    logger.info(f"F1 Score: {f1_score(actual, predicted, average='weighted')*100:.2f}")

    num_classes = 2 if task == 'Informative' else 5
    labels = np.arange(num_classes)

    # confusion matrix
    cm = confusion_matrix(actual, predicted, labels=labels)

    # Calculate percentages
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a heatmap plot
    sns.set(font_scale=1.2)
    sns.heatmap(cm_perc, annot=True, annot_kws={"size": 16}, cmap='Blues', 
                xticklabels=labels, yticklabels=labels, fmt=".2%", cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(f'{plots_path}/cm_{finetuned_model_name}.png', dpi=300)
