from models.basemodel import *
from utils.dataloader import *
from utils.trianer import *
from utils.utils import *
import torch

import click

@click.command()
@click.option(
    "--datasetrootdir",
    "-d",
    help = "path to the dataset"
)
@click.option(
    "--learning-rate",
    "-lr",
    default = 0.01,
    help = "learning rate of the optimizer"
)
@click.option(
    "--batchsize",
    "-b",
    default = 32,
    help = "data batch size"
)
@click.option(
    "--epochs",
    "-e",
    default = 5,
    help = "number of epochs to train the model"
)
@click.option(
    "--savepath",
    "-s",
    default = "./recent_model.pt",
    help = "path to save the checkpoint or trained weights"
)
def trainmod(datasetrootdir : str, lr : float, epochs: int, batchsize : int,
             savepath : str):
    dataset = DataLoader(datasetrootdir)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle = True, batch_size = batchsize)
    model = AudioClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = train_model(model, dataloader, criterion, optimizer, epochs)
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, savepath)