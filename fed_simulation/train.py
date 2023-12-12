from unet import Unet
from loss import DiceLoss
from data import load_data

import numpy as np
from tqdm import tqdm
import logging
import copy
import matplotlib.pyplot as plt
import csv

import torch
import torch.optim as optim


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
server_metrics_list = []


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
    
    

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


def train(global_model, client_model1, client_model2, client_model3,  train_loader_client1, train_loader_client2, train_loader_client3, valloader, epochs):


    for i in range(epochs):
        
        logging.info(f'\n | Global Training Round : {i+1} |\n')
        
        global_model.train()
        client_model1.train()
        client_model2.train()
        client_model3.train()
        
        criterion1 = DiceLoss()
        criterion2 = DiceLoss()
        criterion3 = DiceLoss()
        
        running_loss = 0.0
        
        for b1, b2, b3 in tqdm(zip(train_loader_client1, train_loader_client2, train_loader_client3),total = len(train_loader_client1), desc ="Training"):
            
            # update all local models
            client_model1 = copy.deepcopy(global_model)
            client_model2 = copy.deepcopy(global_model)
            client_model3 = copy.deepcopy(global_model)
            
            optimizer1 = optim.Adam(client_model1.parameters(), lr = 0.001)
            optimizer2 = optim.Adam(client_model2.parameters(), lr = 0.001)
            optimizer3 = optim.Adam(client_model3.parameters(), lr = 0.001)
            
            images1 = b1["img"].to(DEVICE).float()
            labels1 = b1["label"].to(DEVICE)
            optimizer1.zero_grad()
            
            images2 = b2["img"].to(DEVICE).float()
            labels2 = b2["label"].to(DEVICE)
            optimizer2.zero_grad()
            
            images3 = b3["img"].to(DEVICE).float()
            labels3 = b3["label"].to(DEVICE)
            optimizer3.zero_grad()
            
            
            loss1 = criterion1(client_model1(images1), labels1)
            loss2 = criterion2(client_model2(images2), labels2)
            loss3 = criterion3(client_model3(images3), labels3)
    
            loss1.backward()
            loss2.backward()
            loss3.backward()
    
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
    
    
            running_loss += ((loss1.item() + loss2.item() + loss3.item()) / 3)
    
            
            w1 = copy.deepcopy(client_model1.state_dict())
            w2 = copy.deepcopy(client_model2.state_dict())
            w3 = copy.deepcopy(client_model3.state_dict())
            
            global_weights = average_weights([w1, w2, w3])
            
            global_model.load_state_dict(global_weights)

        logging.info(f"Training Epoch {i}: loss {running_loss / len(train_loader_client1)}")
        avg_loss, avg_dice, avg_iou, avg_precision, avg_accuracy, avg_recall = test(global_model, valloader)
        logging.info(f"Evaluation Epoch {i}: loss {avg_loss} / dice {avg_dice} / iou {avg_iou} / precision {avg_precision} / accuracy {avg_accuracy} / recall {avg_recall}")
        server_metrics_list.append({"round": i, "loss": avg_loss, "dice": avg_dice, "iou": avg_iou, "precision": avg_precision, "accuracy": avg_accuracy, "recall": avg_recall})
        
        global_model.train()



if __name__ == '__main__':


    train_loader_client1 = load_data(1, 10)
    train_loader_client2 = load_data(11, 20)
    train_loader_client3 = load_data(21, 30)
    valloader = load_data(31, 33)


    logging.basicConfig(filename='./Running.log', level=logging.INFO, format='%(asctime)s: %(message)s')

    global_model = Unet().to(DEVICE)
    client_model1 = Unet().to(DEVICE)
    client_model2 = Unet().to(DEVICE)
    client_model3 = Unet().to(DEVICE)


    train(global_model, client_model1, client_model2, client_model3,  train_loader_client1, train_loader_client2, train_loader_client3, valloader, 10)

    torch.save(global_model.state_dict(),'COMPLETED.pt')


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








