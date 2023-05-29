import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import argparse
from scripts.dataloader import *
from model import EnzyKR

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_dict(model, pt_files):
    model.load_state_dict(torch.load(pt_files, map_location=device))

def Inference(model, dataloader, device):
    model.eval()

    pred_list = []

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            pred_list.append(pred.view(-1).detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)

    return pred

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True, help='processed dataset')
    parser.add_argument('--model_path', required=True, type=str, help='model path ready to load')
    args = parser.parse_args()

    #data_root = "data"
    DATASET = args.dataset
    model_path = args.model_path

    fpath = os.path.join(DATASET)

    test_set = Data_loader(fpath)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = EnzyKR().to(device)

    load_model_dict(model, model_path)
    preds = Inference(model, test_loader, device)
    for pred in preds:
        print(f"The dG++ are as following: {pred}")


if __name__ == "__main__":
    main()
