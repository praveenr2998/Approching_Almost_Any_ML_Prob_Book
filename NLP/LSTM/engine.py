import torch
import torch.nn as nn

def train(data_loader, model, optimizer, device):
    '''
    This function is used to train the model

    INPUT
    data_loader : Object - The train data loader is passed
    model       : Object - Pytorch model
    optimizer   : Object - Optimizer
    device      : Object - Torch device - CPU/GPU

    RETUNS
    None
    '''
    model.train()

    for data in data_loader:
        reviews = data['review']
        targets = data['target']

        reviews = reviews.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()

        predictions = model(reviews)

        loss = nn.BCEWithLogitsLoss()(
            predictions, targets.view(-1, 1).float()
        )

        loss.backward()
        optimizer.step()

def evaluate(data_loader, model, device):
    '''
    This function is used to evaluate the model

    INPUT
    data_loader : Object - The validation data loader is passed
    model       : Object - Pytorch model
    device      : Object - Torch device - CPU/GPU

    RETUNS
    None
    '''
    final_predictions = []
    final_targets = []

    model.eval()

    with torch.no_grad():
        for data in data_loader:
            reviews = data['review']
            targets = data['target']
            reviews = reviews.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            predictions = model(reviews)

            predictions = predictions.cpu().numpy().tolist()
            targets = data['target'].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)

    return final_predictions, final_targets