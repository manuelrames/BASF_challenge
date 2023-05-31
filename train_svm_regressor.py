import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import os
from statistics import mean

folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

# store MAE for results reporting
trainval_MAEs = []
#val_MAEs = []
test_MAEs = []

for fold in folds:
    print('\nFOLD: %s\n' % fold)

    # import extracted features and health scores
    train_features = np.load(os.path.join('extracted_features/' + fold, 'train_features.npy'))
    train_health_scores = np.load(os.path.join('extracted_features/' + fold, 'train_health_scores.npy'))
    val_features = np.load(os.path.join('extracted_features/' + fold, 'val_features.npy'))
    val_health_scores = np.load(os.path.join('extracted_features/' + fold, 'val_health_scores.npy'))
    test_features = np.load(os.path.join('extracted_features/' + fold, 'test_features.npy'))
    test_health_scores = np.load(os.path.join('extracted_features/' + fold, 'test_health_scores.npy'))
    trainval_features = np.concatenate((train_features, val_features), axis=0)
    trainval_health_scores = np.concatenate((train_health_scores, val_health_scores), axis=0)

    # Fit regression model
    svr_rbf = SVR(kernel="rbf", C=25, gamma='auto', epsilon=0.01)
    svr_rbf.fit(trainval_features, trainval_health_scores)

    # train predictions
    trainval_preds = svr_rbf.predict(trainval_features)
    # val predictions
    #val_preds = svr_rbf.predict(val_features)
    # test predictions
    test_preds = svr_rbf.predict(test_features)

    # calculate MAE between train/val/test predictions and health scores ground truth
    trainval_mae = np.linalg.norm(trainval_health_scores - trainval_preds, ord=1) / trainval_health_scores.size
    trainval_MAEs.append(trainval_mae)
    #val_mae = np.linalg.norm(val_health_scores - val_preds, ord=1) / val_health_scores.size
    #val_MAEs.append(val_mae)
    test_mae = np.linalg.norm(test_health_scores - test_preds, ord=1) / test_health_scores.size
    test_MAEs.append(test_mae)

    print("TrainVal MEA = %f" % trainval_mae)
    print("Test MEA = %f" % test_mae)

# report results
print('TRAINVAL FOLDS MEAN MAE %f' % mean(trainval_MAEs))
#print('VAL FOLDS MEAN MAE %f' % mean(val_MAEs))
print('TEST FOLDS MEAN MAE %f' % mean(test_MAEs))

