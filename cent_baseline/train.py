from unet import Unet
from loss import DiceLoss
from data import load_data

import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import csv

import torch
import torch.optim as optim


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


def train(net, trainloader, valloader, epochs):
    """Train the model on the training set."""
    net.train()
    criterion = DiceLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    for i in range(epochs):
        
        running_loss = 0.0
        
        for batch in tqdm(trainloader, "Training"):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            
            loss = criterion(net(images.to(DEVICE).float()), labels.to(DEVICE))
            
            running_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()

        logging.info(f"Training Epoch {i}: loss {running_loss / len(trainloader)}")
        avg_loss, avg_dice, avg_iou, avg_precision, avg_accuracy, avg_recall = test(net, valloader)
        logging.info(f"Evaluation Epoch {i}: loss {avg_loss} / dice {avg_dice} / iou {avg_iou} / precision {avg_precision} / accuracy {avg_accuracy} / recall {avg_recall}")
        server_metrics_list.append({"round": i, "loss": avg_loss, "dice": avg_dice, "iou": avg_iou, "precision": avg_precision, "accuracy": avg_accuracy, "recall": avg_recall})
        
        net.train()





if __name__ == '__main__':


    trainloader = load_data(1, 30)
    valloader = load_data(31, 33)


    logging.basicConfig(filename='./Running.log', level=logging.INFO, format='%(asctime)s: %(message)s')

    model = Unet().to(DEVICE)


    train(model, trainloader, valloader, 10)

    torch.save(model.state_dict(),'COMPLETED.pt')


    # Specify the file path
    csv_file_path = 'output.csv'

    # Open the CSV file in write mode
    with open(csv_file_path, 'w', newline='') as csv_file:
        # Create a CSV writer object

        csv_writer = csv.DictWriter(csv_file, fieldnames=server_metrics_list[0].keys())

        # Write the header
        csv_writer.writeheader()

        # Write the data
        csv_writer.writerows(server_metrics_list)
    logging.info(f'Data has been saved to {csv_file_path}')



    server_epochs = np.arange(len(server_metrics_list))
    server_dice_values = [metrics["dice"] for metrics in server_metrics_list]
    server_iou_values = [metrics["iou"] for metrics in server_metrics_list]


    server_loss_values = [metrics["loss"] for metrics in server_metrics_list]
    server_precision_values = [metrics["precision"] for metrics in server_metrics_list]
    server_accuracy_values = [metrics["accuracy"] for metrics in server_metrics_list]
    server_recall_values = [metrics["recall"] for metrics in server_metrics_list]




    plt.figure(figsize=(10, 6))
    plt.plot(server_epochs, server_dice_values, label="Dice Coefficient")
    plt.plot(server_epochs, server_loss_values, label="Loss")
    plt.xlabel("Round")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.savefig("metrics_plot_loss.png")


    plt.figure(figsize=(10, 6))
    plt.plot(server_epochs, server_precision_values, label="Precision")
    plt.plot(server_epochs, server_iou_values, label="IoU")
    plt.plot(server_epochs, server_accuracy_values, label="Accuracy")
    plt.plot(server_epochs, server_recall_values, label="Recall")

    plt.xlabel("Round")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.savefig("metrics_plot.png")








