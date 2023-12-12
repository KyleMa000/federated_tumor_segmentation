import flwr as fl
import torch

from unet import Unet
from loss import DiceLoss
from data import load_data

import os
import csv
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
server_metrics_list = []


def test(net, testloader):
    """Validate the model on the test set."""
    net.eval()
    criterion = DiceLoss()
    dice, loss, iou, precision, accuracy, recall = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(testloader, "Testing"):
            images = batch["img"]
            labels = batch["label"]

            outputs = net(images.to(DEVICE).float())
            
            # Calculate Soft Dice coefficient
            current_loss = criterion(outputs, labels.to(DEVICE)).item()

            dice += 1 - current_loss

            # Calculate loss
            loss += current_loss
            
            outputs = torch.sigmoid(outputs)
            
            outputs = torch.flatten(outputs)  # Flatten
            labels = torch.flatten(labels)

            # Calculate IOU (Jaccard Index)
            intersection = torch.sum(torch.logical_and(outputs > 0.5, labels.to(DEVICE) > 0.5))
            union = torch.sum(torch.logical_or(outputs > 0.5, labels.to(DEVICE) > 0.5))
            current_iou = intersection.item() / union.item() if union.item() > 0 else 0.0
            iou += current_iou

            # Calculate precision, accuracy, and recall
            predicted_labels = (outputs > 0.5).float()
            true_positives = torch.sum(torch.logical_and(predicted_labels == 1, labels.to(DEVICE) == 1)).item()
            true_negatives = torch.sum(torch.logical_and(predicted_labels == 0, labels.to(DEVICE) == 0)).item()
            false_positives = torch.sum(torch.logical_and(predicted_labels == 1, labels.to(DEVICE) == 0)).item()
            false_negatives = torch.sum(torch.logical_and(predicted_labels == 0, labels.to(DEVICE) == 1)).item()

            current_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            current_accuracy = (true_positives + true_negatives) / len(outputs)
            current_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

            precision += current_precision
            accuracy += current_accuracy
            recall += current_recall
    
    avg_dice = dice / len(testloader)
    avg_iou = iou / len(testloader)
    avg_precision = precision / len(testloader)
    avg_accuracy = accuracy / len(testloader)
    avg_recall = recall / len(testloader)
    avg_loss = loss / len(testloader)
    
    net.train()
    
    return avg_loss, avg_dice, avg_iou, avg_precision, avg_accuracy, avg_recall



def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    
    


    net = Unet().to(DEVICE)
    valloader = load_data(31,33)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, avg_dice, avg_iou, avg_precision, avg_accuracy, avg_recall = test(net, valloader)    
    server_metrics_list.append({"round": server_round, "loss": loss, "dice": avg_dice, "iou": avg_iou, "precision": avg_precision, "accuracy": avg_accuracy, "recall": avg_recall})


    # Specify the file path
    csv_file_path = 'output_round.csv'

    # Check if the file already exists
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)

    # Open the CSV file in write mode
    with open(csv_file_path, 'w', newline='') as csv_file:
        # Create a CSV writer object

        csv_writer = csv.DictWriter(csv_file, fieldnames=server_metrics_list[0].keys())

        # Write the header
        csv_writer.writeheader()

        # Write the data
        csv_writer.writerows(server_metrics_list)
    print(f'Data has been saved to {csv_file_path}')

    return loss, {"dice": avg_dice, "iou": avg_iou, "precision": avg_precision, "accuracy": avg_accuracy, "recall": avg_recall}


def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


params = get_parameters(Unet())

# Define strategy
strategy = fl.server.strategy.FedAvg(min_available_clients=3,
                                     min_fit_clients = 3,
                                     min_evaluate_clients=3,
                                     evaluate_fn=evaluate,
                                     initial_parameters=fl.common.ndarrays_to_parameters(params)
                                     )


# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)



# Plot and save progresses

server_epochs = np.arange(len(server_metrics_list))
server_dice_values = [metrics["dice"] for metrics in server_metrics_list]
server_iou_values = [metrics["iou"] for metrics in server_metrics_list]

server_loss_values = [metrics["loss"] for metrics in server_metrics_list]
server_precision_values = [metrics["precision"] for metrics in server_metrics_list]
server_accuracy_values = [metrics["accuracy"] for metrics in server_metrics_list]
server_recall_values = [metrics["recall"] for metrics in server_metrics_list]

plt.figure(figsize=(10, 6))
plt.plot(server_epochs, server_dice_values, label="Server Dice Coefficient")
plt.plot(server_epochs, server_loss_values, label="Server Loss")
plt.xlabel("Round")
plt.ylabel("Metric Value")
plt.legend()
plt.savefig("server_metrics_plot_loss.png")

plt.figure(figsize=(10, 6))
plt.plot(server_epochs, server_precision_values, label="Server Precision")
plt.plot(server_epochs, server_iou_values, label="Server IoU")
plt.plot(server_epochs, server_accuracy_values, label="Server Accuracy")
plt.plot(server_epochs, server_recall_values, label="Server Recall")

plt.xlabel("Round")
plt.ylabel("Metric Value")
plt.legend()
plt.savefig("server_metrics_plot.png")