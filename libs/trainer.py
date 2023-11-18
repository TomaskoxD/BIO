##############################################
#                                            #
#    BIO                                     #
#                                            #
#    Trainer and validator for the model.    #
#                                            #
#    Author: Tomáš Ondrušek (xondru18)       #
#            Peter Ďurica   (xduric05)       #
#                                            #
##############################################

import time
import torch
from common.progress.bar import Bar
import sys
sys.path.append('../common')
import numpy as np
import pandas as pd

class Trainer:
    """
    Trainer

    Class to train the model. It trains the model and saves the best model based on the validation loss. It trains model on 3 sets of images, all from the same eye but in different color channels. It uses the Trainer class from PyTorch to train the model.

    Args:
        model: Model to train.
        optimizer: Optimizer for training.
        criterion: Loss function.
        loss_weights: Weights for loss function.
        epochs: Number of epochs.
    
    Returns:
        None.
    """
    def __init__(self, model, optimizer, criterion, loss_weights, epochs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_weights = loss_weights
        self.epochs = epochs

    def train_step(self, train_loader, epoch):
        self.model.train()
        epoch_loss = 0.0
        iters_per_epoch = len(train_loader)
        bar = Bar('Processing {} Epoch -> {} / {}'.format('train', epoch+1, self.epochs), max=iters_per_epoch)
        bar.check_tty = False

        for step, (imagesA, imagesB, imagesC, labels) in enumerate(train_loader):
            start_time = time.time()

            torch.set_grad_enabled(True)

            imagesA = imagesA.cuda()
            imagesB = imagesB.cuda()
            imagesC = imagesC.cuda()

            labels = labels.cuda()
            labels = labels.squeeze(1)

            out_A, out_B, out_C, out_F, combine = self.model(imagesA, imagesB, imagesC)

            loss_x = self.criterion(out_A, labels)
            loss_y = self.criterion(out_B, labels)
            loss_z = self.criterion(out_C, labels)
            loss_c = self.criterion(out_F, labels)
            loss_f = self.criterion(combine, labels)

            lossValue = (
                self.loss_weights[0] * loss_x +
                self.loss_weights[1] * loss_y +
                self.loss_weights[2] * loss_z +
                self.loss_weights[3] * loss_c +
                self.loss_weights[4] * loss_f
            )

            self.optimizer.zero_grad()
            lossValue.backward()
            self.optimizer.step()

            epoch_loss += lossValue.item()
            end_time = time.time()
            batch_time = end_time - start_time

            bar_str = '{} / {} | Time: {batch_time:.1f} mins | Loss: {loss:.4f} '
            bar.suffix = bar_str.format(
                step + 1, iters_per_epoch, batch_time=batch_time * (iters_per_epoch - step) / 60,
                loss=lossValue.item()
            )
            bar.next()

        epoch_loss = epoch_loss / iters_per_epoch
        bar.finish()
        return epoch_loss
class Validator:
    """
    Validator

    Class to validate the model. It validates the model and saves the best model based on the validation loss. It validates model on 3 sets of images, all from the same eye but in different color channels. It uses the Validator class from PyTorch to validate the model.

    Args:
        model: Model to validate.
        criterion: Loss function.

    Returns:
        None.
    """
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def validate(self, val_loader):
        self.model.eval()
        epoch_loss = 0
        iters_per_epoch = len(val_loader)
        bar = Bar('Processing {}'.format('validation'), max=iters_per_epoch)

        for step, (imagesA, imagesB, imagesC, labels) in enumerate(val_loader):
            start_time = time.time()

            imagesA = imagesA.cuda()
            imagesB = imagesB.cuda()
            imagesC = imagesC.cuda()

            labels = labels.cuda()
            labels = labels.squeeze(1)

            _, _, _, _, outputs = self.model(imagesA, imagesB, imagesC)
            with torch.no_grad():
                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item()

            end_time = time.time()

            batch_time = end_time - start_time
            bar_str = '{} / {} | Time: {batch_time:.2f} mins'
            bar.suffix = bar_str.format(step + 1, len(val_loader), batch_time=batch_time * (iters_per_epoch - step) / 60)
            bar.next()

        epoch_loss = epoch_loss / iters_per_epoch
        bar.finish()
        return epoch_loss

def save_output(label_test_file, dataPRED, args, save_file):
    """
    Save output.

    Function to save the output of the model. It saves the output of the model in a csv file.

    Args:
        label_test_file: Label file for test data.
        dataPRED: Predicted data.
        args: Arguments.
        save_file: File to save the output.

    Returns:
        None.
    """
    label_list = args.label_idx
    n_class = len(label_list)
    datanpPRED = np.squeeze(dataPRED.cpu().numpy())
    df_tmp = pd.read_csv(label_test_file)
    image_names = df_tmp["image"].tolist()

    result = {label_list[i]: datanpPRED[:, i] for i in range(n_class)}
    result['image_name'] = image_names
    out_df = pd.DataFrame(result)

    name_older = ['image_name']
    for i in range(n_class):
        name_older.append(label_list[i])
    out_df.to_csv(save_file, columns=name_older)


