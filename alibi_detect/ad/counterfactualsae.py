import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import kld
from tensorflow.keras.models import Model
from typing import Callable, Dict, List, Tuple, Union
from alibi_detect.models.autoencoder import CFAE
from alibi_detect.models.trainer import trainer
from alibi_detect.models.losses import loss_cf
from alibi_detect.utils.prediction import predict_batch
from alibi_detect.base import (BaseDetector, FitMixin, ThresholdMixin,
                               adversarial_prediction_dict, adversarial_correction_dict)

logger = logging.getLogger(__name__)


class DenseHidden(tf.keras.Model):

    def __init__(self, model: tf.keras.Model, hidden_layer: int, output_dim: int, hidden_dim: int = None) -> None:
        """
        Dense layer that extracts the feature map of a hidden layer in a model and computes
        output probabilities over that layer.

        Parameters
        ----------
        model
            tf.keras classification model.
        hidden_layer
            Hidden layer from model where feature map is extracted from.
        output_dim
            Output dimension for softmax layer.
        hidden_dim
            Dimension of optional additional dense layer.
        """
        super(DenseHidden, self).__init__()
        self.partial_model = Model(inputs=model.inputs, outputs=model.layers[hidden_layer].output)
        for layer in self.partial_model.layers:  # freeze model layers
            layer.trainable = False
        self.hidden_dim = hidden_dim
        if hidden_dim is not None:
            self.dense_layer = Dense(hidden_dim, activation=tf.nn.relu)
        self.output_layer = Dense(output_dim, activation=tf.nn.softmax)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.partial_model(x)
        x = Flatten()(x)
        if self.hidden_dim is not None:
            x = self.dense_layer(x)
        return self.output_layer(x)


class CounterfactualsAE(FitMixin):

    def __init__(self,
                 threshold: float = None,
                 ae: tf.keras.Model = None,
                 model: tf.keras.Model = None,
                 encoder_net: tf.keras.Sequential = None,
                 decoder_net: tf.keras.Sequential = None,
                 model_hl: List[tf.keras.Model] = None,
                 hidden_layer_kld: dict = None,
                 w_model_hl: list = None,
                 temperature: float = 1.,
                 loss_type: str = 'cce', # 'cce' or 'kld'
                 data_type: str = None
                 ) -> None:
        """
        Autoencoder (AE) based adversarial detector.

        Parameters
        ----------
        threshold
            Threshold used for adversarial score to determine adversarial instances.
        ae
            A trained tf.keras autoencoder model if available.
        model
            A trained tf.keras classification model.
        encoder_net
            Layers for the encoder wrapped in a tf.keras.Sequential class if no 'ae' is specified.
        decoder_net
            Layers for the decoder wrapped in a tf.keras.Sequential class if no 'ae' is specified.
        model_hl
            List with tf.keras models for the hidden layer K-L divergence computation.
        hidden_layer_kld
            Dictionary with as keys the hidden layer(s) of the model which are extracted and used
            during training of the AE, and as values the output dimension for the hidden layer.
        w_model_hl
            Weights assigned to the loss of each model in model_hl.
        temperature
            Temperature used for model prediction scaling.
            Temperature <1 sharpens the prediction probability distribution.
        data_type
            Optionally specifiy the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to infer threshold using `infer_threshold`.')

        self.threshold = threshold
        self.model = model
        for layer in self.model.layers:  # freeze model layers
            layer.trainable = False

        # check if model can be loaded, otherwise initialize AE model
        if isinstance(ae, tf.keras.Model):
            self.ae = ae
        elif isinstance(encoder_net, tf.keras.Sequential) and isinstance(decoder_net, tf.keras.Sequential):
            self.ae = CFAE(encoder_net, decoder_net, self.model)  # define AE model
        else:
            raise TypeError('No valid format detected for `ae` (tf.keras.Model) '
                            'or `encoder_net` and `decoder_net` (tf.keras.Sequential).')

        # intermediate feature map outputs for KLD and loss weights
        self.hidden_layer_kld = hidden_layer_kld
        if isinstance(model_hl, list):
            self.model_hl = model_hl
        elif isinstance(hidden_layer_kld, dict):
            self.model_hl = []
            for hidden_layer, output_dim in hidden_layer_kld.items():
                self.model_hl.append(DenseHidden(self.model, hidden_layer, output_dim))
        else:
            self.model_hl = None
        self.w_model_hl = w_model_hl
        if self.w_model_hl is None and isinstance(self.model_hl, list):
            self.w_model_hl = list(np.ones(len(self.model_hl)))

        self.temperature = temperature
        self.loss_type = loss_type

    def fit(self,
            X: np.ndarray,
            loss_fn: tf.keras.losses = loss_cf,
            w_model: float = 1.,
            w_recon: float = 0.,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            epochs: int = 20,
            batch_size: int = 128,
            verbose: bool = True,
            log_metric: Tuple[str, "tf.keras.metrics"] = None,
            callbacks: tf.keras.callbacks = None,
            preprocess_fn: Callable = None
            ) -> None:
        """
        Train Adversarial AE model.

        Parameters
        ----------
        X
            Training batch.
        loss_fn
            Loss function used for training.
        w_model
            Weight on model prediction loss term.
        w_recon
            Weight on MSE reconstruction error loss term.
        optimizer
            Optimizer used for training.
        epochs
            Number of training epochs.
        batch_size
            Batch size used for training.
        verbose
            Whether to print training progress.
        log_metric
            Additional metrics whose progress will be displayed if verbose equals True.
        callbacks
            Callbacks used during training.
        preprocess_fn
            Preprocessing function applied to each training batch.
        """
        # train arguments
        args = [self.ae, loss_fn, X]
        kwargs = {
            'optimizer': optimizer,
            'epochs': epochs,
            'batch_size': batch_size,
            'verbose': verbose,
            'log_metric': log_metric,
            'callbacks': callbacks,
            'preprocess_fn': preprocess_fn,
            'cf_model': self.model,
            'loss_type': self.loss_type,
            'loss_fn_kwargs': {
                'model': self.model,
                'temperature': self.temperature,
                'loss_type': self.loss_type
            }
        }

        # train
        trainer(*args, **kwargs)

    def predict(self, X: np.ndarray, y_t: np.ndarray, batch_size: int = int(1e10)):
        """
        Predict whether instances are adversarial instances or not.

        Parameters
        ----------
        X
            Batch of instances.
        batch_size
            Batch size used when computing scores.
        return_instance_score
            Whether to return instance level adversarial scores.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the adversarial predictions and instance level adversarial scores.
        """

        return self.ae(X, y_t).numpy()
