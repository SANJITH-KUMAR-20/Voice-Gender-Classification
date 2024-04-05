from models.basemodel import *
from utils.dataloader import *
from utils.trianer import *
from utils.utils import *
import torch


def predict(model_path:str, data) -> str:
    model = AudioClassifier()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    aud = process_audio(data)
    pred = model(aud.unsqueeze(0))
    pred = torch.nn.Softmax(dim = 1)(pred)
    pred = pred.detach().numpy().astype(int).tolist()
    if pred == [1,0]:
        return "female"
    else:
        return "male"