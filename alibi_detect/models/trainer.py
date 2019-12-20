import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Callable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras.losses import kld
import os
from tensorflow.keras.callbacks import ModelCheckpoint

def trainer(model: tf.keras.Model,
            ancilla_model: tf.keras.Model,
            loss_fn: tf.keras.losses,
            X_train: np.ndarray,
            y_train: np.ndarray = None,
            validation_data: tuple = None,
            adversarial_data: tuple = None,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss_fn_kwargs: dict = None,
            epochs: int = 20,
            batch_size: int = 64,
            buffer_size: int = 1024,
            verbose: bool = True,
            log_metric:  Tuple[str, "tf.keras.metrics"] = None,
            log_metric_val: Tuple[str, Callable] = None,
            log_dir: str = None,
            callbacks: tf.keras.callbacks = None) -> None:  # TODO: incorporate callbacks + LR schedulers
    """
    Train TensorFlow model.

    Parameters
    ----------
    model
        Model to train.
    loss_fn
        Loss function used for training.
    X_train
        Training batch.
    y_train
        Training labels.
    optimizer
        Optimizer used for training.
    loss_fn_kwargs
        Kwargs for loss function.
    epochs
        Number of training epochs.
    batch_size
        Batch size used for training.
    buffer_size
        Maximum number of elements that will be buffered when prefetching.
    verbose
        Whether to print training progress.
    log_metric
        Additional metrics whose progress will be displayed if verbose equals True.
    callbacks
        Callbacks used during training.
    """
    # create datase
    if y_train is None:  # unsupervised model
        train_data = X_train
    else:
        train_data = (X_train, y_train)
    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    train_data = train_data.shuffle(buffer_size=buffer_size).batch(batch_size)
    n_minibatch = int(np.ceil(X_train.shape[0] / batch_size))

    if validation_data is not None:
        X_val, y_val = validation_data
        if y_val is None:
            validation_data = X_val
        else:
            validation_data = (X_val, y_val)
        validation_data = tf.data.Dataset.from_tensor_slices(validation_data)
        validation_data = validation_data.shuffle(buffer_size=buffer_size).batch(len(X_val))

    if adversarial_data is not None:
        X_adv, y_adv = adversarial_data
        if y_adv is None:
            adversarial_data = X_adv
        else:
            adversarial_data = (X_adv, y_adv)
        adversarial_data = tf.data.Dataset.from_tensor_slices(adversarial_data)
        adversarial_data = adversarial_data.shuffle(buffer_size=buffer_size).batch(len(X_adv))

    train_loss, test_loss = [], []
    test_scores, test_accs, test_f1s, test_recs, test_precs, test_cms = [], [], [], [], [], []
    adv_scores = []
    # iterate over epochs
    for epoch in range(epochs):
        if verbose:
            pbar = tf.keras.utils.Progbar(n_minibatch, 1)

        # iterate over the batches of the dataset
        for step, train_batch in enumerate(train_data):

            if y_train is None:
                X_train_batch = train_batch
            else:
                X_train_batch, y_train_batch = train_batch

            with tf.GradientTape() as tape:
                preds = model(X_train_batch)
                if y_train is None:
                    ground_truth = X_train_batch
                else:
                    ground_truth = y_train_batch

                # compute loss
                if tf.is_tensor(preds):
                    args = [ground_truth, preds]
                else:
                    args = [ground_truth] + list(preds)

                if loss_fn_kwargs:
                    loss = loss_fn(*args, **loss_fn_kwargs)
                else:
                    loss = loss_fn(*args)

                if model.losses:  # additional model losses
                    loss += sum(model.losses)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if verbose:
                loss_val = loss.numpy()
                if loss_val.shape != (batch_size,) and loss_val.shape:
                    add_mean = np.ones((batch_size - loss_val.shape[0],)) * loss_val.mean()
                    loss_val = np.r_[loss_val, add_mean]
                pbar_values = [('loss', loss_val)]
                if log_metric is not None:
                    log_metric[1](ground_truth, preds)
                    pbar_values.append((log_metric[0], log_metric[1].result().numpy()))
                pbar.add(1, values=pbar_values)
        train_loss.append(loss_val)

        # Validation
        if adversarial_data is not None and validation_data is not None:
            # prepare advesarial test data

            step_adv, adv_batch = next(enumerate(adversarial_data))

            if y_adv is None:
                X_adv_batch = adv_batch
            else:
                X_adv_batch, y_adv_batch = adv_batch

            with tf.GradientTape() as tape:
                preds_adv = model(X_adv_batch)
                if y_adv is None:
                    ground_truth_adv = X_adv_batch
                else:
                    ground_truth_adv = y_adv_batch

                # compute val loss
                if tf.is_tensor(preds_adv):
                    args = [X_adv_batch, preds_adv]
                else:
                    args = [X_adv_batch] + list(preds_adv)

                if loss_fn_kwargs:
                    loss_valid = loss_fn(*args, **loss_fn_kwargs)
                else:
                    loss_valid = loss_fn(*args)

            # prepare validation data
            step_val, val_batch = next(enumerate(validation_data))

            X_val_batch = val_batch

            with tf.GradientTape() as tape:
                preds_val = model(X_val_batch)
                ground_truth_val = X_val_batch

                # compute val loss
                if tf.is_tensor(preds_val):
                    args = [X_val_batch, preds_val]
                else:
                    args = [X_val_batch] + list(preds_val)

                if loss_fn_kwargs:
                    loss_valid = loss_fn(*args, **loss_fn_kwargs)
                else:
                    loss_valid = loss_fn(*args)

            loss_valid_val = loss_valid.numpy()
            if loss_valid_val.shape != (len(X_val),) and loss_valid_val.shape:
                add_mean_valid = np.ones((len(X_val) - loss_valid_val.shape[0],)) * loss_valid_val.mean()
                loss_valid_val = np.r_[loss_valid_val, add_mean_valid]
                if verbose:
                    pbar_values = [('loss_valid', loss_valid_val)]
            if log_metric_val is not None:
                train_scores = score(X_train_batch.numpy(), model, ancilla_model)
                threshold = infer_threshold(train_scores)
                adv_score = score(X_adv_batch.numpy(), model, ancilla_model)
                y_preds_adv = (adv_score > threshold).astype(int)

                acc = accuracy_score(y_adv_batch.numpy(), y_preds_adv)
                f1 = f1_score(y_adv_batch.numpy(), y_preds_adv)
                prec = precision_score(y_adv_batch.numpy(), y_preds_adv)
                rec = recall_score(y_adv_batch.numpy(), y_preds_adv)
                cm = confusion_matrix(y_adv_batch.numpy(), y_preds_adv)

                best_model_path = os.path.join(log_dir, 'best.ckpt')
                if len(test_accs) == 0:
                    max_acc = 0
                else:
                    max_acc = max(test_accs)
                if acc > max_acc:
                    print('Accuracy improved from {} to {}. Saving model in {}'.format(np.round(max_acc, 4),
                                                                                       np.round(acc, 4),
                                                                                       best_model_path))
                    model.save_weights(best_model_path)

                test_accs.append(acc)
                test_f1s.append(f1)
                test_precs.append(prec)
                test_recs.append(rec)
                test_cms.append(cm)
                adv_scores.append(adv_score)
                if verbose:
                    pbar_values.append(('detection_acc', acc))
                    pbar_values.append(('detection_f1', f1))
                    pbar.add(1, values=pbar_values)
                test_loss.append(loss_valid_val)

                #epoch_model_path = os.path.join(log_dir, 'model_epoch_{}.ckpt'.format(epoch))
                #print('Saving last model')
                #model.save_weights(epoch_model_path)

    if log_dir is not None:
        df_scores, df_loss, df_adv_test_scores = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(data=adv_scores)
        df_scores['acc'] = test_accs
        df_scores['f1'] = test_f1s
        df_scores['rec'] = test_recs
        df_scores['prec'] = test_precs

        df_loss['train'] = train_loss
        df_loss['test'] = test_loss

        df_adv_test_scores = df_adv_test_scores.T
        df_adv_test_scores['labels'] = y_adv_batch

        df_scores.to_csv(os.path.join(log_dir, 'scores.csv'), index=False)
        df_loss.to_csv(os.path.join(log_dir, 'losses.csv'), index=False)
        df_adv_test_scores.to_csv(os.path.join(log_dir, 'adv_scores.csv'), index=False)


def infer_threshold(adv_score,
    threshold_perc: float = 90.
    ):
    """
    Update threshold by a value inferred from the percentage of instances considered to be
    adversarial in a sample of the dataset.

    Parameters
    ----------
    X
    Batch of instances.
    threshold_perc
    Percentage of X considered to be normal based on the adversarial score.
    """
    # update threshold
    threshold = np.percentile(adv_score, threshold_perc)
    return threshold


def score(X: np.ndarray, vae, model, nb_samples=2) -> np.ndarray:
    """
    Compute adversarial scores.

    Parameters
    ----------
    X
    Batch of instances to analyze.

    Returns
    -------
    Array with adversarial scores for each instance in the batch.
    """
    # sample reconstructed instances
    X_samples = np.repeat(X, nb_samples, axis=0)
    X_recon = vae(X_samples)

    # model predictions
    y = model(X_samples)
    y_recon = model(X_recon)

    # KL-divergence between predictions
    kld_y = kld(y, y_recon).numpy().reshape(-1, nb_samples)
    adv_score = np.mean(kld_y, axis=1)
    return adv_score