import numpy as np
import tensorflow as tf
from typing import Tuple, Callable
from sklearn.metrics import accuracy_score

def trainer(model: tf.keras.Model,
            loss_fn: tf.keras.losses,
            X_train: np.ndarray,
            y_train: np.ndarray = None,
            validation_data: tuple = (None, None),
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss_fn_kwargs: dict = None,
            epochs: int = 20,
            batch_size: int = 64,
            buffer_size: int = 1024,
            verbose: bool = True,
            log_metric:  Tuple[str, "tf.keras.metrics"] = None,
            log_metric_val: Tuple[str, Callable] = None,
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

    if validation_data[0] is not None:
        if validation_data[1] is None:
            validation_data = validation_data[0]
        validation_data = tf.data.Dataset.from_tensor_slices(validation_data)

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

            if validation_data != (None, None):
                model.infer_threshold(X_train[:2000], threshold_perc=90.)
                if isinstance(validation_data, tuple):
                    preds_val = model.predict(validation_data[0])
                    ground_truth_val = validation_data[1]
                else:
                    preds_val = model.predict(validation_data)
                    ground_truth_val = validation_data

                if log_metric_val is not None:
                    log_metric_val[1](ground_truth_val, preds_val)
                    pbar_values.append((log_metric_val[0], log_metric_val[1].result().numpy()))
            pbar.add(1, values=pbar_values)


def advesarial_accuracy(ground_truth_val, preds_val):
    score_val = preds_val['data']['instance_score']
    preds_val = preds_val['data']['is_adversarial']

    acc = accuracy_score(ground_truth_val, preds_val)

    return acc
