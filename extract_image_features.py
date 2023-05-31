import os
import torch
from torch import nn
from torchvision import models
import numpy as np

from train_image_regressor import PlantAssessmentDataset, data_transforms

# create folder where to store features
features_folder = 'extracted_features'
os.makedirs(features_folder, exist_ok=True)

folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
best_trial = ['trial4', 'trial5', 'trial1', 'trial4', 'trial5'] # best trial following test loss for comparison

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for i, fold in enumerate(folds):
    # create fold subfolder
    os.makedirs(os.path.join(features_folder, fold), exist_ok=True)

    # load the previously trained model
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, 256, bias=True),
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(256, 128, bias=True),
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(128, 64, bias=True),
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(64, 32, bias=True),
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(32, 1, bias=True),
    )

    model_ft.load_state_dict(torch.load(os.path.join('runs_resnet18/' + fold + '/' + best_trial[i], 'model_resnet18_besttestloss.pth')))
    model_ft.eval()
    model_ft.to(device)

    # delete the decision layers (nn.Linear and nn.Dropouts) to leave features as final outputs
    model_ft.fc = nn.Identity()

    # prepare datasets and dataloaders
    data_dir = os.path.join('data_folds', fold)
    image_datasets = {x: PlantAssessmentDataset(os.path.join(data_dir, x),
                                                data_transforms['test'])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                 shuffle=True, num_workers=0)
                  for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}


    for phase in ['train', 'val', 'test']:
        # variable that store features
        phase_features = None
        phase_health_scores = None

        for inputs, health_scores in dataloaders[phase]:
            inputs = inputs.to(device)

            # store health scores into output tensor
            if phase_health_scores is None:
                phase_health_scores = health_scores
            else:
                phase_health_scores = torch.cat((phase_health_scores, health_scores), 0)

            with torch.no_grad():
                outputs = model_ft(inputs).cpu().detach()

                # store extracted features into output tensor
                if phase_features is None:
                    phase_features = outputs
                else:
                    phase_features = torch.cat((phase_features, outputs), 0)

        # convert to numpy arrays for easier manipulation later
        phase_health_scores = phase_health_scores.numpy()
        phase_features = phase_features.numpy()

        # save features and health scores
        np.save(os.path.join(features_folder + '/' + fold, phase + '_features'), phase_features)
        np.save(os.path.join(features_folder + '/' + fold, phase + '_health_scores'), phase_health_scores)

        #torch.save(phase_features, os.path.join(features_folder + '/' + fold, phase + '_features.pt'))
        #torch.save(phase_health_scores, os.path.join(features_folder + '/' + fold, phase + '_health_scores.pt'))