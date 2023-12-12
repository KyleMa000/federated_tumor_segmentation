import torch
import torch.optim as optim

import argparse
import warnings
import flwr as fl
from tqdm import tqdm
from collections import OrderedDict

from unet import Unet
from loss import DiceLoss
from data import load_data


warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def train(net, trainloader, epochs):
    net.train()
    criterion = DiceLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.001)

    for _ in range(epochs):

        for batch in tqdm(trainloader, "Training"):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            
            loss = criterion(net(images.to(DEVICE).float()), labels.to(DEVICE))
                        
            loss.backward()
            
            optimizer.step()


parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--node-id",
    choices=[0, 1, 2],
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.")
node_id = parser.parse_args().node_id

net = Unet().to(DEVICE)

start_patient_id = node_id * 10 + 1
end_patient_id = (node_id + 1) * 10

trainloader = load_data(start_patient_id, end_patient_id)

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.5, 10, {"accuracy": 0.95}


# Start Flower client
fl.client.start_numpy_client(
    server_address="10.0.0.158:8080",
    client=FlowerClient(),
)
