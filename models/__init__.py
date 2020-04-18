import torch
import torch.nn as nn

from models.basic import BasicModel

def get_model(args):
    if args.model == 'basic':
        model = BasicModel()
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise 'No such model'
        
    return model, criterion
