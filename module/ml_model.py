"""
This class provides unsupervised deep machine learning model used for
predicting the phase of RIS.
"""
from module import system
from module import data_generator as gen
from module import data_transform as transform

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Concatenate,
    Flatten,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.activations import sigmoid, relu
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, regularizers
import pickle
import time
import matplotlib.pyplot as plt


class DNN:
    def __init__(self, config):
        """
        Initialization
        """
        self.batch_size = config["DNN"]["batch_size"]
        self.learning_rate = config["DNN"]["learning_rate"]
        self.num_epochs = config["DNN"]["num_epochs"]
        self.l2 = config["DNN"]["l2"]
        self.drop_out_rate = config["DNN"]["drop_out_rate"]

        # Number of tensors at each hidden layers
        self.num_tensors = config["DNN"]["num_tensors"]
        self.num_output = config["system_model"]["RIS"]["num_phase"]

        # Number of cluster head
        self.K = config["system_model"]["CH"]["K"]

        # path for saving the DNN objects and tf.keras model for reuse
        self.path_to_model = config["path"]["path_to_model"]
        # loss value during training process, this will be assigned during
        # training process
        self.loss_for_training = None
        self.loss_for_validation = None
        # this should be assigned before calling training method
        self.data_transformer = transform.DataNormalization()
        # tf.keras DNN model
        self.model = None

    def construct_DNN_model(self) -> None:
        """
        Construct the DNN model. The number of hidden layers and tensors are
        specified in self.num_tensors

        Inputs of DNN:
            3 inputs includes:
                F (Column vector)
                R (Column vector)
                A (Matrix)

        Outputs: Normalized phases by 2 * pi, which is between 0 and 1.
        """
        # build a DNN with 3 inputs and 1 output
        F = Input(shape=(self.K, 1), name="f")
        R = Input(shape=(self.K, 1), name="r")
        A = Input(shape=(self.K, self.num_output), name="a")

        # Get the real and im part of A
        A_real = tf.math.real(A)
        A_im = tf.math.imag(A)

        # Flatten
        A_real = Flatten()(A_real)
        A_im = Flatten()(A_im)

        F_flat = Flatten()(F)
        R_flat = Flatten()(R)

        # Concatenate
        layer = Concatenate(axis=-1)([A_real, A_im, F_flat, R_flat])

        # Multi-layer perceptron
        for num_tensor in self.num_tensors:
            layer = Dense(num_tensor, activation=relu)(layer)
            # layer = Dropout(0.7)(layer)
            layer = BatchNormalization()(layer)

        phi = Dense(self.num_output, activation=sigmoid)(layer)

        self.model = Model(inputs=[F, R, A], outputs=phi)

    def cal_loss(self, F: tf.float32, R: tf.float32, A: tf.complex64) -> tf.float32:
        """
        Calculate the loss function for a given input F, R and A

        Args:
            F (tf.float32): [F.shape = (None, K, 1)]
            R (tf.float32): [R.shape = (None, K, 1)]
            A (tf.complex64): [A.shape = (None, K, num_outputs)]

        Returns:
            tf.float32: [The loss value]
        """
        assert (
            self.data_transformer.mean_R_db != None
        ), "The data_transformer should be assigned before training."

        # Feed forward to find phi
        phi = self.model([F, R, A])
        phi = tf.cast(phi, dtype=tf.complex64)
        # get the column vector phi by multiply phi with 2 * pi * 1j
        vector_phi = tf.exp(1j * 2 * np.pi * phi[..., None])

        # Get the true value of F and R from the normalized input in dB
        F = 10.0 ** (
            (self.data_transformer.sigma_F_db * F + self.data_transformer.mean_F_db)
            / 10.0
        )
        R = 10.0 ** (
            (self.data_transformer.sigma_R_db * R + self.data_transformer.mean_R_db)
            / 10.0
        )

        # spectral efficiency
        se = (
            tf.math.log(
                1
                + tf.math.add(
                    tf.cast(F, dtype=tf.float32),
                    tf.multiply(
                        tf.cast(R, dtype=tf.float32),
                        tf.math.square(
                            tf.abs(
                                tf.matmul(tf.cast(A, dtype=tf.complex64), vector_phi)
                            )
                        ),
                    ),
                )
            )
            / np.log(2)
        )

        # We want to maximize the sum of ergodic spectral efficiency
        # Axis = 1 means that we want to keep the data along the side
        # of batch dimension
        objective_loss = -tf.math.reduce_sum(se, axis=1)

        return tf.reduce_mean(objective_loss)

    def fit(
        self,
        F_train: np.array,
        R_train: np.array,
        A_train: np.array,
        F_val: np.array,
        R_val: np.array,
        A_val: np.array,
        cascaded_data: bool = True,
    ) -> None:
        """
        Args:
            F_train (np.array): [Training data]
            R_train (np.array): [Training data]
            A_train (np.array): [Training data]
            F_val (np.array): [Validation data]
            R_val (np.array): [Validation data]
            A_val (np.array): [Validation data]
            cascaded_data (bool, optional): [True - to specify that the input
                data A_train only contains the cascaded angle and need to be
                transformed before feeding into the batch training.]
                Defaults to True.
        """

        assert (
            self.model != None
        ), "The DNN model has not been built. Call construct_DNN_model first"

        F_train_normalize, R_train_normalize = self.data_transformer.transform(
            F_train, R_train
        )
        F_val_normalize, R_val_normalize = self.data_transformer.transform(F_val, R_val)

        # optimizer
        opt = optimizers.Adam(learning_rate=self.learning_rate)
        # Number of updates for one epoch
        num_updates_train = int(F_train.shape[0] / self.batch_size)
        num_updates_val = int(F_val.shape[0] / self.batch_size)
        # "ratio": (int) in one epoch, loss on validation is calculated once
        # per "ratio" iterations on train
        ratio = int(num_updates_train / num_updates_val)

        @tf.function
        def step(F, R, A):
            """
            Perform one step of updating weights using gradients
            """
            with tf.GradientTape() as tape:
                # calculate the loss and gradients
                loss = self.cal_loss(F, R, A)
                grads = tape.gradient(loss, self.model.trainable_variables)
                # update the variables of DNN
                opt.apply_gradients(zip(grads, self.model.trainable_variables))

                return loss

        # start training
        loss_for_training = np.array(self.num_epochs * [0.0])
        loss_for_val = np.array(self.num_epochs * [0.0])

        print("Start training process")
        for epoch in range(self.num_epochs):
            start_time = time.time()
            # loss values for each epoch
            loss_train_in_epoch = np.array(num_updates_train * [0.0])
            loss_val_in_epoch = np.array(num_updates_val * [0.0])
            # update weights and loss value for each batch
            for i in range(0, num_updates_train):
                # get on batch of training data
                start = i * self.batch_size
                end = start + self.batch_size
                # we transform the data in each batch to avoid being out of memory
                if cascaded_data:
                    A_train_batch = gen.generate_angle_data_from_cascasded_angle(
                        A_train[start:end], self.num_output
                    )
                else:
                    A_train_batch = A_train[start:end]
                # take a step, we feed the normalized F and normalized R in dB
                loss_train_in_epoch[i] = step(
                    F_train_normalize[start:end],
                    R_train_normalize[start:end],
                    A_train_batch,
                ).numpy()

                # for every "ratio" iterations, compute loss on one batch
                # of validation set
                if i % ratio == 0:
                    # get one batch of validation data
                    start = int(i / ratio) * self.batch_size
                    end = start + self.batch_size
                    if cascaded_data:
                        A_val_batch = gen.generate_angle_data_from_cascasded_angle(
                            A_val[start:end], self.num_output
                        )
                    else:
                        A_val_batch = A_val[start:end]
                    # calculate the loss on a batch of validation data
                    loss_val_in_epoch[int(i / ratio)] = self.cal_loss(
                        F_val_normalize[start:end],
                        R_val_normalize[start:end],
                        A_val_batch,
                    ).numpy()

            loss_for_training[epoch] = np.mean(loss_train_in_epoch)
            loss_for_val[epoch] = np.mean(loss_val_in_epoch)

            end_time = time.time()
            print(
                "Epoch {} - Loss on train = {: .4f} - Loss on validation = {: .4f} - Time excuted = {: .2f} seconds".format(
                    epoch,
                    loss_for_training[epoch],
                    loss_for_val[epoch],
                    end_time - start_time,
                )
            )
            # Assign the loss values into the class and save model, plot the
            # losses every 10 epoches
            if epoch % 10 == 0 and epoch != 0:
                self.loss_for_training = loss_for_training[:epoch]
                self.loss_for_val = loss_for_val[:epoch]
                self.save_model()
                plt.plot(loss_for_training[:epoch], color="black", label="Training")
                plt.plot(loss_for_val[:epoch], color="blue", label="Validation")
                plt.legend()
                plt.title("Loss values over epochs on training and validation")
                plt.xlabel("Epoch")
                plt.ylabel("Loss values")
                plt.show()

    def save_model(self) -> None:
        """
        Save the data_transformer and tf.keras model.
        """
        with open(self.path_to_model + "/data_transformer.pickle", "wb") as f:
            pickle.dump(self.data_transformer, f)
            print(
                "The data_transformer has been saved in ",
                self.path_to_model + "/data_transformer.pickle",
            )

        with open(self.path_to_model + "/loss_for_training.pickle", "wb") as f:
            pickle.dump(self.loss_for_training, f)
            print(
                "The loss values for training has been saved in ",
                self.path_to_model + "/loss_for_training.pickle",
            )

        with open(self.path_to_model + "/loss_for_val.pickle", "wb") as f:
            pickle.dump(self.loss_for_val, f)
            print(
                "The loss values for validation has been saved in ",
                self.path_to_model + "/loss_for_val.pickle",
            )

        # Save the model
        self.model.save(self.path_to_model)
        print("The tf.keras DNN model has been saved into ", self.path_to_model)

    def load_model(self, path: str = None) -> None:
        """
        Load the data_transformer and the tf.keras model from the disk.

        Args:
            path (str): [Path to the ml model]. Defaults is None.
                If path is None, load model from the path in config.
                Else, load model from the input path.
        """
        if path == None:
            path = self.path_to_model
        else:
            self.path_to_model = path
        # load data_transformer
        with open(path + "/data_transformer.pickle", "rb") as read:
            self.data_transformer = pickle.load(read)
        # load the loss training values
        with open(path + "/loss_for_training.pickle", "rb") as read:
            self.loss_for_training = pickle.load(read)
        # load the loss values on validation
        with open(path + "/loss_for_val.pickle", "rb") as read:
            self.loss_for_val = pickle.load(read)
        # loss the tf.keras model
        self.model = tf.keras.models.load_model(path)

    def predict(self, large_scale_coef: system.LargeScaleCoef) -> np.array:
        """
        Predict the phase with the input Large Scale Coef.

        Args:
            large_scale_coef (system.LargeScaleCoef): Large scale coefs
                estimated by the BS

        Returns:
            np.array: [a column vector of phases in range 0 to 2 * pi]
        """
        assert large_scale_coef.F.shape == (self.K, 1)
        assert large_scale_coef.R.shape == (self.K, 1)
        assert large_scale_coef.A.shape == (self.K, self.num_output)

        F_db_normalized, R_db_normalized = self.data_transformer.transform(
            large_scale_coef.F, large_scale_coef.R
        )
        return (
            2
            * np.pi
            * self.model.predict(
                [
                    np.array([F_db_normalized]),
                    np.array([R_db_normalized]),
                    np.array([large_scale_coef.A]),
                ],
            ).reshape((self.num_output, 1))
        )
