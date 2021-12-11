import torch
import numpy as np
import random

import checkpoint
from dataset import VehicleDataset
from model import CNN
from plot import Plotter
import csv
import pdb

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def predictions(logits):
    """
    Compute the predictions from the model.
    Inputs:
        - logits: output of our model based on some input, tensor with shape=(batch_size, num_classes)
    Returns:
        - pred: predictions of our model, tensor with shape=(batch_size)
    """
    batch_size = logits.size(0)
    num_classes = logits.size(1)
    pred = torch.zeros(logits.size(0))
    for i in range(batch_size):
        max_index = 0
        for j in range(num_classes):
            if logits[i][j] > logits[i][max_index]:
                max_index = j
        pred[i] = max_index
    return pred


def accuracy(y_true, y_pred):
    """
    Compute the accuracy given true and predicted labels.
    Inputs:
        - y_true: true labels, tensor with shape=(num_examples)
        - y_pred: predicted labels, tensor with shape=(num_examples)
    Returns:
        - acc: accuracy, float
    """
    num_accurate = 0
    num_examples = y_pred.size(0)
    for i in range(num_examples):
        if y_true[i]==y_pred[i]:
            num_accurate += 1
    return float(num_accurate)/float(num_examples)


def _train_epoch(train_loader, model, criterion, optimizer):
    """
    Train the model for one iteration through the train set.
    """
    for i, (X, y) in enumerate(train_loader):
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()


def _evaluate_epoch(plotter, train_loader, val_loader, model, criterion, epoch):
    """
    Evaluates the model on the train set.
    """
    stat = []
    data_loader_list = [train_loader, val_loader]
    for i in range(len(data_loader_list)):
        data_loader = data_loader_list[i]
        y_true, y_pred, running_loss = evaluate_loop(data_loader, model, criterion)
        total_loss = np.sum(running_loss) / y_true.size(0)
        total_acc = accuracy(y_true, y_pred)
        stat += [total_acc, total_loss]
        if i == 1 and epoch == 10:
            with open("result.csv", "a+") as f:
                csv_writer = csv.writer(f)
                for j in range(len(y_pred)):
                    csv_writer.writerow([float(y_pred[j])])
    plotter.stats.append(stat)
    plotter.log_cnn_training(epoch)
    plotter.update_cnn_training_plot(epoch)


def evaluate_loop(data_loader, model, criterion=None):
    model.eval()
    y_true, y_pred, running_loss = [], [], []
    for X, y in data_loader:
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            if criterion is not None:
                running_loss.append(criterion(output, y).item() * X.size(0))
    model.train()
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    return y_true, y_pred, running_loss


def train(config, dataset, model):
    # Data loaders
    train_loader, val_loader = dataset.train_loader, dataset.val_loader

    if 'use_weighted' not in config:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0,20.0]))
        
    learning_rate = config['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Attempts to restore the latest checkpoint if exists
    print('Loading model...')
    force = config['ckpt_force'] if 'ckpt_force' in config else False
    model, start_epoch, stats = checkpoint.restore_checkpoint(model, config['ckpt_path'], force=force)

    # Create plotter
    plot_name = config['plot_name'] if 'plot_name' in config else 'CNN'
    plotter = Plotter(stats, plot_name)

    # Evaluate the model
    _evaluate_epoch(plotter, train_loader, val_loader, model, criterion, start_epoch)

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config['num_epoch']):
        # Train model on training set
        _train_epoch(train_loader, model, criterion, optimizer)

        # Evaluate model on training and validation set
        _evaluate_epoch(plotter, train_loader, val_loader, model, criterion, epoch + 1)

        # Save model parameters
        checkpoint.save_checkpoint(model, epoch + 1, config['ckpt_path'], plotter.stats)

    print('Finished Training')

    # Save figure and keep plot open
    plotter.save_cnn_training_plot()
    plotter.hold_training_plot()


if __name__ == '__main__':
    # define config parameters for training
    config = {
        'dataset_path': 'data',
        'batch_size': 4,
        'ckpt_path': 'checkpoints/cnn',  # directory to save our model checkpoints
        'num_epoch': 10,                 # number of epochs for training
        'learning_rate': 1e-4,           # learning rate
    }
    # create dataset
    dataset = VehicleDataset(config['batch_size'], config['dataset_path'])
    # create model
    model = CNN()
    # train our model on dataset
    train(config, dataset, model)
