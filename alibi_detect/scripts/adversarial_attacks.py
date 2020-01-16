import numpy as np
import pandas as pd
import tensorflow
import foolbox
from sklearn.model_selection import ParameterGrid
import joblib

datasets = ['fashion_mnist']

def clean_grid(pars_list):
    clean_list = []
    for params in pars_list:
        if 'epsilon' in params.keys() and 'epsilon_iter' in params.keys():
            if params['epsilon'] > params['epsilon_iter']:
                clean_list.append(params)
            else:
                pass
        else:
            clean_list.append(params)
    return clean_list

grid = [{'attack': ['carlini'],
        'binary_search_steps': [10, 20],
        'max_iterations': [1000],
        'confidence': [0, 0.1, 0.2, 0.5, 0.8],
        'learning_rate': [0.05, 5e-3],
        'initial_const': [1e-2, 1e-3],
        'abort_early': [True]
         },
        {'attack': ['pfgsm'],
        'binary_search': [True] ,
        'random_start': [False],
        'return_early': [True],
        'iterations':[10, 20],
        'epsilon':[0.1, 0.2, 0.3, 0.5, 1., 2.],
        'stepsize': [0.1, 0.2, 0.5, 1.]
         },
        {'attack': ['fgsm'],
         'binary_search': [True],
         'epsilons':[0.1, 0.2, 0.3, 0.5, 1., 2.],
         'stepsize': [0.01, 0.02, 0.05, 0.1],
         'iterations': [10, 20],
         'random_start': [False],
         'return_early': [True]
         },
        {'attack': ['rpgsm'],
         'binary_search': [True],
         'epsilon': [0.1, 0.2, 0.3, 0.5, 1., 2.],
         'stepsize': [0.01, 0.02, 0.05, 0.1],
         'iterations': [10, 20],
         'random_start': [False],
         'return_early': [True]
         }]

params_list = list(ParameterGrid(grid))

def load_dataset(dataset):

    if dataset == 'fashion_mnist':
        train, test = tensorflow.keras.datasets.fashion_mnist.load_data()
        X_train, y_train = train
        X_test, y_test = test

        X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
        X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
        # X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
        # X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')
        y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
        y_test = tensorflow.keras.utils.to_categorical(y_test, 10)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    elif dataset == 'mnist':
        train, test = tensorflow.keras.datasets.mnist.load_data()
        X_train, y_train = train
        X_test, y_test = test

        X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
        X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
        # X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
        # X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')
        y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
        y_test = tensorflow.keras.utils.to_categorical(y_test, 10)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    elif dataset == 'cifar10':
        pass

    model = tensorflow.keras.models.load_model(
        '/home/gio/adversarial_vae/adversarial-vae/models/classifier/{}/model/model.h5'.format(dataset))

    return X_train, X_test, model


def performe_attack(dataset, **kwargs):
    X_train, X_test, model = load_dataset(dataset)
    # X_train, X_test = X_train[:10], X_test[:10]
    fmodel = foolbox.models.KerasModel(model, bounds=(0, 1))

    if kwargs['attack'] == 'carlini':
        fb_attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel)

    elif kwargs['attack'] == 'pfgsm':
        fb_attack = foolbox.attacks.ProjectedGradientDescent(fmodel)

    if kwargs['attack'] == 'fgsm':
        fb_attack = foolbox.attacks.FGSM(fmodel)

    if kwargs['attack'] == 'rpgsm':
        fb_attack = foolbox.attacks.RandomPGD(fmodel)

    if kwargs['attack'] == 'deepfool':
        pass

    kwargs = {k: v for k, v in kwargs.items() if k != 'attack'}

    batch_size = 10000
    nb_samples = X_train.shape[0]
    nb_batches = np.ceil(nb_samples / batch_size).astype(int)

    adv_batches = []
    for i in range(nb_batches):
        X_batch = X_train[i * batch_size: (i+1) * batch_size]
        labels_batch = np.argmax(fmodel.forward(X_batch), axis=1)
        X_adv_batch = fb_attack(X_batch, labels_batch, **kwargs)
        adv_batches.append(X_adv_batch)

    X_adv_train = np.concatenate(adv_batches, axis=0)

    labels_test = np.argmax(fmodel.forward(X_test), axis=1)
    X_adv_test = fb_attack(X_test, labels_test, **kwargs)

    return X_adv_train, X_adv_test


def main():

    counter = 0
    for dataset in datasets:
        for kwargs in params_list:
            print('Dataset', dataset)
            print('Attack', kwargs)
            print('Attacking ...')
            X_adv_train, X_adv_test = performe_attack(dataset, **kwargs)
            dict_to_save = {'params': kwargs, 'X_adv_train': X_adv_train, 'X_adv_test': X_adv_test}
            save_path = '/home/gio/datasets/adversarial/{}/params_{}.joblib'.format(dataset, counter)
            print('Attack completed. Saving in', save_path)
            joblib.dump(dict_to_save, save_path)
            print('Attack saved!')


if __name__=="__main__":
    main()
    print('Done!')
