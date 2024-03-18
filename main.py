import os
import numpy as np
import pandas as pd
from aeon.datasets import load_classification

from utils.tools import create_directory
from utils.constants import datasets

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def fit_classifier(all_labels, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=16):
    nb_classes = len(np.unique(all_labels))
    # Create Classifier --------------------------------------------------------
    if classifier_name == "FCN" or classifier_name == "ResNet":
        input_shape = (X_train.shape[1], X_train.shape[2])
    elif classifier_name == "lstm_dcnn" or classifier_name == "MLSTM_FCN":
        input_shape = (X_train.shape[1], X_train.shape[2])
    else:
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    # Call Classifier ----------------------------------------------------------
    classifier = create_classifier(classifier_name, input_shape, nb_classes, verbose=True)
    # Train Classifier ----------------------------------------------------------
    if X_val is None:
        classifier.fit(X_train, y_train, None, None, epochs, batch_size)
    else:
        classifier.fit(X_train, y_train, X_val, y_val, epochs, batch_size)
    return classifier


def create_classifier(classifier_name, input_shape, nb_classes, verbose=False):
    if classifier_name == "Disjoint_CNN":
        from classifiers import Disjoint_CNN
        return Disjoint_CNN.Classifier_Disjoint_CNN(sub_output_directory, input_shape, nb_classes, verbose)


# Problem Setting -----------------------------------------------------------------------------------------------------
ALL_Results = pd.DataFrame()
ALL_Results_list = []
problem_index = 0
data_path = os.getcwd() + '/multivariate_ts/'
# Hyper-Parameter Setting ----------------------------------------------------------------------------------------------
classifier_name = "Disjoint_CNN"  # Choose the classifier name from aforementioned List
epochs = 500
Resample = 1  # Set to '1' for default Train and Test Sets, and '30' for running on all resampling
# ----------------------------------------------------------------------------------------------------------------------

for problem in datasets:
    # Load Data --------------------------------------------------------------------------------------------------------
    output_directory = os.getcwd() + '/Results_' + classifier_name + '/' + problem + '/'
    create_directory(output_directory)
    print("[Main] Problem: {}".format(problem))
    itr_result = [problem]
    # load --------------------------------------------------------------------------
    # set data folder
    X_train, Y_train = load_classification(problem,  split="train")
    X_test, Y_test = load_classification(problem,  split="test")

    all_data = np.vstack((X_train, X_test))
    all_labels = np.hstack((Y_train, Y_test))
    all_indices = np.arange(len(all_data))

    sub_output_directory = output_directory + str(1) + '/'
    create_directory(sub_output_directory)
    # Default Train and Test Set
    x_train = X_train
    x_test = X_test
    y_train = Y_train
    y_test = Y_test
    
    # Making Consistent with Keras Output -------------------------------------------------
    all_labels_new = np.concatenate((y_train, y_test), axis=0)
    print("[Main] All labels: {}".format(np.unique(all_labels_new)))
    tmp = pd.get_dummies(all_labels_new).values
    y_train = tmp[:len(y_train)]
    y_test = tmp[len(y_train):]

    # Making Consistent with Keras Input ---------------------------------------------------
    if classifier_name == "FCN" or classifier_name == "ResNet" or classifier_name == "MLSTM_FCN":
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[1])
    else:
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # classifier-----------------------------------------------------------------
    # Dynamic Batch-size base on Data
    if 'EigenWorms' in problem or 'DuckDuck' in problem:
        batch_size = 1
    else:
        # batch_size = np.ceil(x_train.shape[0] / (8 * (np.max(y_train.shape[1]) + 1)))
        batch_size = 8
        
    val_index = np.random.randint(0, np.int(x_train.shape[0]), np.int(x_train.shape[0] / 10), dtype=int)
    x_val = x_train[val_index, :]
    y_val = y_train[val_index, :]

    classifier = fit_classifier(all_labels_new, x_train, y_train, x_val, y_val, epochs, batch_size)
    metrics_test, conf_mat = classifier.predict(x_test, y_test, best=True)
    metrics_test2, conf_mat2 = classifier.predict(x_test, y_test, best=False)

    metrics_test['train/val/test/test2'] = 'test'
    metrics_test2['train/val/test/test2'] = 'test2'
    metrics = pd.concat([metrics_test, metrics_test2]).reset_index(drop=True)
    
    print("[Main] Problem: {}".format(problem))
    print(metrics.head())

    metrics.to_csv(sub_output_directory + 'classification_metrics.csv')
    np.savetxt(sub_output_directory + 'confusion_matrix.csv', conf_mat, delimiter=",")
    itr_result.append(metrics.accuracy[0])
    itr_result.append(metrics.accuracy[1])
    sub_output_directory = []

    if len(ALL_Results_list) == 0:
        ALL_Results_list = np.hstack((ALL_Results_list, itr_result))
    else:
        ALL_Results_list = np.vstack((ALL_Results_list, itr_result))

    problem_index = problem_index + 1

ALL_Results = pd.DataFrame(ALL_Results_list)
ALL_Results.to_csv(os.getcwd() + '/Results_' + classifier_name + '/'+'All_results1.csv')
