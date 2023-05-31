from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import os
import copy

from PIL import Image
import re

cudnn.benchmark = True
plt.ion()   # interactive mode

class PlantAssessmentDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(root)))
        with open('challenge_anno.txt') as f:
            self.score_labels_txt = f.readlines()

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # get the plant health label
        health_score = [re.search('\t(.*)\n', x).group(1) for x in self.score_labels_txt if self.imgs[idx] in x]
        #health_score = torch.tensor(int(health_score[0]), dtype=torch.int64)
        health_score = torch.tensor(int(health_score[0]), dtype=torch.float32)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, health_score

    def __len__(self):
        return len(self.imgs)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(344),
        transforms.RandomRotation(degrees=180, fill=255),
        transforms.RandomResizedCrop(344, scale=(0.80, 1)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(344),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
            transforms.Resize(344),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}

if __name__ == "__main__":

    folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
    for fold in folds:
        print('\nFOLD: %s\n' % fold)

        for num_trial in range(1, 6):

            # log metrics to tensorboard
            with SummaryWriter() as writer:
                writer = SummaryWriter('runs_resnet18/' + fold + '/trial' + str(num_trial))

                data_dir = os.path.join('data_folds', fold)
                image_datasets = {x: PlantAssessmentDataset(os.path.join(data_dir, x),
                                                          data_transforms[x])
                                  for x in ['train', 'val', 'test']}
                dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                                             shuffle=True, num_workers=0)
                              for x in ['train', 'val', 'test']}
                dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


                ######################################################################
                # Visualize a few images
                # ^^^^^^^^^^^^^^^^^^^^^^
                # Let's visualize a few training images so as to understand the data
                # augmentations.

                def imshow(inp, title=None):
                    """Display image for Tensor."""
                    inp = inp.numpy().transpose((1, 2, 0))
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    inp = std * inp + mean
                    inp = np.clip(inp, 0, 1)
                    plt.imshow(inp)
                    if title is not None:
                        plt.title(title)
                    plt.pause(0.001)  # pause a bit so that plots are updated


                # Get a batch of training data
                #inputs, health_scores = next(iter(dataloaders['train']))

                # Make a grid from batch
                #out = torchvision.utils.make_grid(inputs)

                #imshow(out, title=[x.item() for x in health_scores])


                def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
                    since = time.time()

                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_min_loss = 10000000

                    for epoch in range(num_epochs):
                        print(f'Epoch {epoch}/{num_epochs - 1}')
                        print('-' * 10)

                        # Each epoch has a training and validation phase
                        for phase in ['train', 'val']:
                            if phase == 'train':
                                model.train()  # Set model to training mode
                            else:
                                model.eval()   # Set model to evaluate mode

                            running_loss = 0.0
                            #running_corrects = 0

                            # Iterate over data.
                            for inputs, health_scores in dataloaders[phase]:
                                inputs = inputs.to(device)
                                #health_scores = health_scores.to(device)
                                health_scores = torch.unsqueeze(health_scores, dim=1).to(device)

                                # zero the parameter gradients
                                optimizer.zero_grad()

                                # forward
                                # track history if only in train
                                with torch.set_grad_enabled(phase == 'train'):
                                    outputs = model(inputs)
                                    loss = criterion(outputs, health_scores)
                                    # log loss to tensorboard
                                    writer.add_scalar(phase + "_loss", loss, epoch)

                                    # backward + optimize only if in training phase
                                    if phase == 'train':
                                        loss.backward()
                                        optimizer.step()

                                # statistics
                                running_loss += loss.item() * inputs.size(0)
                                #running_corrects += torch.sum(preds == labels.data)
                            if phase == 'train':
                                scheduler.step()

                            epoch_loss = running_loss / dataset_sizes[phase]
                            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

                            print(f'{phase} Loss: {epoch_loss:.4f}')

                            # deep copy the model
                            if phase == 'val' and epoch_loss < best_min_loss:
                                best_min_loss = epoch_loss
                                best_model_wts = copy.deepcopy(model.state_dict())

                        print()

                    # test phase
                    model.eval()  # Set model to evaluate mode
                    test_running_loss = 0
                    for inputs, health_scores in dataloaders['test']:
                        inputs = inputs.to(device)
                        #health_scores = health_scores.to(device)
                        health_scores = torch.unsqueeze(health_scores, dim=1).to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            loss = criterion(outputs, health_scores)

                        # statistics
                        test_running_loss += loss.item() * inputs.size(0)
                    test_loss = test_running_loss / dataset_sizes['test']

                    time_elapsed = time.time() - since
                    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    print(f'Best val Loss: {best_min_loss:4f}')
                    print(f'TEST Loss: {test_loss:.4f}')
                    writer.add_text('TEST Loss', str(test_loss), num_epochs)

                    writer.add_hparams({'optimizer': 'SGD', 'epochs': num_epochs, 'lr': optimizer.defaults['lr'], 'momentum': optimizer.defaults['momentum'],
                                        'step_size': scheduler.step_size, 'gamma': scheduler.gamma, 'bs': 8, 'loss_function': 'L1Loss'},
                                      {'best_val_loss': best_min_loss, 'best_test_loss': test_loss})

                    writer.flush()

                    # load best model weights
                    model.load_state_dict(best_model_wts)
                    return model


            ######################################################################
            # Visualizing the model predictions
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #
            # Generic function to display predictions for a few images
            #

            def visualize_model(model, split_name, num_images=12):
                was_training = model.training
                model.eval()
                images_so_far = 0
                fig = plt.figure()

                with torch.no_grad():
                    for i, (inputs, labels) in enumerate(dataloaders[split_name]):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs)
                        outputs = torch.squeeze(outputs)
                        #_, preds = torch.max(outputs, 1)

                        for j in range(inputs.size()[0]):
                            images_so_far += 1
                            ax = plt.subplot(num_images//2, 2, images_so_far)
                            ax.axis('off')
                            ax.set_title(f'HR: {int(labels[j])} Predicted: {outputs[j]}')
                            imshow(inputs.cpu().data[j])

                            if images_so_far == num_images:
                                model.train(mode=was_training)
                                return
                    model.train(mode=was_training)

            ######################################################################
            # Finetuning the ConvNet
            # ----------------------
            #
            # Load a pretrained model and reset final fully connected layer.
            #

            model_ft = models.resnet18(weights='IMAGENET1K_V1')
            num_ftrs = model_ft.fc.in_features
            # Here the size of each output sample is set to 1.
            #model_ft.fc = nn.Linear(num_ftrs, 100)
            #model_ft.classifier[1] = nn.Linear(num_ftrs, 1,  bias=True)
            """model_ft.fc = nn.Sequential(
                                    nn.Dropout(p=0.4, inplace=True),
                                    nn.Linear(num_ftrs, 1024,  bias=True),
                                    nn.Dropout(p=0.4, inplace=True),
                                    nn.Linear(1024, 512,  bias=True),
                                    nn.Dropout(p=0.4, inplace=True),
                                    nn.Linear(512, 256,  bias=True),
                                    nn.Dropout(p=0.4, inplace=True),
                                    nn.Linear(256, 128,  bias=True),
                                    nn.Dropout(p=0.4, inplace=True),
                                    nn.Linear(128, 64,  bias=True),
                                    nn.Dropout(p=0.4, inplace=True),
                                    nn.Linear(64, 32,  bias=True),
                                    nn.Dropout(p=0.4, inplace=True),
                                    nn.Linear(32, 1,  bias=True),
            )"""
            model_ft.fc = nn.Sequential(
                                    nn.Dropout(p=0.4, inplace=True),
                                    nn.Linear(num_ftrs, 256,  bias=True),
                                    nn.Dropout(p=0.4, inplace=True),
                                    nn.Linear(256, 128,  bias=True),
                                    nn.Dropout(p=0.4, inplace=True),
                                    nn.Linear(128, 64,  bias=True),
                                    nn.Dropout(p=0.4, inplace=True),
                                    nn.Linear(64, 32,  bias=True),
                                    nn.Dropout(p=0.4, inplace=True),
                                    nn.Linear(32, 1,  bias=True),
            )

            model_ft = model_ft.to(device)

            #criterion = nn.MSELoss()
            criterion = nn.L1Loss()

            # Observe that all parameters are being optimized
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0005, momentum=0.9)

            # Decay LR by a factor of 0.5 every 4 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.5)

            model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                   num_epochs=25)

            # store best model for each fold
            torch.save(model_ft.state_dict(), os.path.join('runs_resnet18/', fold + '/trial' + str(num_trial), 'model_resnet18_bestvalloss.pth'))

            writer.flush()
            writer.close()

            ######################################################################
            #

            #visualize_model(model_ft, 'val')
            #visualize_model(model_ft, 'test')

            plt.ioff()
            plt.show()
