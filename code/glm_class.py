## This script contains the GLM and GLM_CV class definitions and utility functions for fitting generalized linear models (GLM) with Tnesorflow 2. It was developed with the following:

# numpy version 1.21.6
# sklearn version 1.0.2
# scipy version 1.7.3
# tensorflow version 2.8.2
# keras version 2.8.0
# matplotlib version 3.2.2

## The code works most efficiently on a GPU. Please refer to the associated iPython notebook "Tutorial_for_using_GLM_class.ipynb" for how to use it (https://github.com/sytseng/GLM_Tensorflow_2/tree/main/tutorial).

## Written by Shih-Yi Tseng from the Harvey Lab at Harvard Medical School, with special acknowledgements to Matthias Minderer and Selmaan Chettih. Last updated: 2022/08/23

## References:
# Tseng, S.-Y., Chettih, S.N., Arlt, C., Barroso-Luque, R., and Harvey, C.D. (2022). Shared and specialized coding across posterior cortical areas for dynamic navigation decisions. Neuron 110, 2484–2502.e16
# Minderer, M., Brown, K.D., and Harvey, C.D. (2019). The spatial structure of neural encoding in mouse posterior cortex during navigation. Neuron 102, 232–248.e11


import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tensorflow as tf
from scipy.linalg import block_diag
from sklearn.model_selection import GroupKFold, KFold

#### GLM class ####


class GLM:
    def __init__(
        self,
        activation="exp",
        loss_type="poisson",
        regularization="elastic_net",
        lambda_series=10.0 ** np.linspace(-1, -8, 30),
        l1_ratio=0.0,
        smooth_strength=0.0,
        optimizer="adam",
        learning_rate=1e-3,
        momentum=0.5,
        min_iter_per_lambda=100,
        max_iter_per_lambda=10**4,
        num_iter_check=100,
        convergence_tol=1e-6,
    ):
        """
        GLM class

        Train generalized linear models (GLM) for a range of regularization values on training data,
        select models with validation data, and evaluate / make prediction on test data with the fitted/selected weights and intercepts.
        Fit Y with multiple responses simultaneously.

        Model: Y = activation(X * w + w0) + noise
        X: n_samples x n_features
        Y: n_samples x n_responses
        w: n_features x n_responses
        w0: n_responses

        Use following combinations of activation and loss_type for common model types:
        Gaussian: 'linear' + 'gaussian'; Poisson: 'exp' + 'poisson'; Logistic: 'sigmoid' + 'binominal'
        Or create you own combinations

        Input parameters::
        activation: {'linear', 'exp', 'sigmoid', 'relu', 'softplus'}, default = 'exp'
        loss_type: {'gaussian', 'poisson', 'binominal'}, default = 'poisson'
        regularization: {'elastic_net', 'group_lasso'}, default = 'elastic_net'
        lambda_series: list or ndarray of a series of regularization strength (lambda), in descending order,
                       default = 10.0 ** np.linspace(-1, -8, 30)
        l1_ratio: L1 ratio for elastic_net regularization (l1_ratio = 1. is Lasso, l1_ratio = 0. is ridge), default = 0.
        smooth_strength: strength for smoothness penalty, default = 0.
        optimizer: {'adam', 'sgdm'}, default = 'adam'
        learning_rate: learning rate for optimizer, default = 1e-3
        momentum: momentum for sgdm optimizer, default = 0.5
        min_iter_per_lambda: minimal iterations for each lambda, default = 100
        max_iter_per_lambda: maximal iterations for each lambda, default = 10000
        num_iter_check: number of iterations for checking convergence (delta loss is averaged over this number), default = 100
        convergence_tol: convergence criterion, complete fitting when absolute average delta loss over past num_iter_check
                         iterations is smaller than convergence_tol*average null deviance of the data, default = 1e-6

        Attributes::
        Key attributes you might look up after fitting and model selection:
          selected_w0, selected_w, selected_lambda, selected_lambda_ind, selected_frac_dev_expl_val
        In addition, examining loss_trace may help you determine your hyperparameters for fitting,
        such as learning rate, convergence_tol, etc.

        List of all attributes:
        act_func: activation function based on input activation type (tensorflow function)
        loss_func: loss function based on input loss type (tensorflow function)
        reg_func: regularization function based on input regularization type (tensorflow function)
        fitted: if the model has been fitted, bool
        selected: if model selection has been performed, bool
        n_features: number of features seen during fit (X.shape[1])
        n_responses: number of responses seen during fit (Y.shape[1])
        n_lambdas: number of regularization strengths (lambda_series.shape[0])
        feature_group_size: size of each group for regularization = 'group_lasso', list of len = n_groups,
                            provided by the user as input to fit method when regularization = 'group_lasso' or smooth_strength > 0.
        group_matrix: group matrix used in fitting for regularization = 'group_lasso', a matrix that converts from a n_group vector
                      to an n_expanded_group vector for scaling the groups differently, tensor of shape (n_groups, n_features)
        prior_matrix: prior matrix used in fitting for smooth_strength > 0., a block-diagonal matrix containing [-1, 2, 1] on the
                      diagonal for expanded features, tensor of shape (n_features, n_features)
        w_series: fitted intercepts and weights for all lambdas,
                  list of len n_lambdas as [[w0, w] for lambda 1, [w0, w] for lambda 2, ..., etc.]
        loss_trace: list of training loss on each iteration during fitting
        lambda_trace: list of lambda value for each iteration during fitting
        selected_w0: intercepts for selected models, ndarray of shape (n_responses,)
        selected_w: weights for selected models, ndarray of shape (n_features, n_responses)
        selected_lambda: lambda values for each response for selected models, ndarray of shape (n_responses,)
        selected_lambda_ind: indices of lambdas for each response for selected models, ndarray of shape (n_responses,)
        selected_frac_dev_expl_val: fraction deviance explained evaluated on the validation data for selected models

        Methods::
        fit(X, Y, [initial_w0, initial_w, feature_group_size, verbose]): fit GLM to training data
        select_model(X_val, Y_val, [min_lambda, make_fig]): select model using validation data after fit is called
        predict(X): returns prediction on input data X using selected models after select_model is called
        evaluate(X, Y, [make_fig]): compute fraction deviance explained on input data X, Y using selected models
                                    after select_model is called
        """

        self.activation = activation
        self.loss_type = loss_type
        self.regularization = regularization
        self.lambda_series = np.sort(np.array(lambda_series))[
            ::-1
        ]  # make sure to fit starts with the largest regularization
        self.n_lambdas = self.lambda_series.shape[0]
        self.l1_ratio = l1_ratio
        self.smooth_strength = smooth_strength
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.min_iter_per_lambda = min_iter_per_lambda
        self.max_iter_per_lambda = max_iter_per_lambda
        self.num_iter_check = num_iter_check
        self.convergence_tol = convergence_tol

        self.fitted = False
        self.selected = False

        # select activation function
        if activation == "linear":
            self.act_func = lambda z: z
        elif activation == "exp":
            self.act_func = lambda z: tf.math.exp(z)
        elif activation == "relu":
            self.act_func = lambda z: tf.nn.relu(z)
        elif activation == "softplus":
            self.act_func = lambda z: tf.math.softplus(z)
        elif activation == "sigmoid":
            self.act_func = lambda z: tf.math.sigmoid(z)

        # select regularization
        if self.regularization == "elastic_net":
            self.reg_func = lambda w: (
                (1.0 - self.l1_ratio) * tf.reduce_sum(tf.square(w) / 2.0)
                + self.l1_ratio * tf.reduce_sum(tf.abs(w))
            )

        elif self.regularization == "group_lasso":
            self.reg_func = lambda w, grouping_mat, feature_group_size: tf.reduce_sum(
                tf.sqrt(tf.matmul(grouping_mat, tf.square(w)))
                * tf.sqrt(feature_group_size)[:, None]
            )

        if self.smooth_strength > 0.0:
            self.smooth_reg_func = lambda w, P: self.smooth_strength * tf.einsum(
                "ij,ik,kj->", w, P, w
            )

        # select loss function
        if np.logical_and(self.loss_type == "poisson", self.activation == "exp"):
            # use pre-activation value because tf.nn.log_poisson_loss takes log-inputs
            self.loss_func = lambda Y, Y_hat, Y_act: tf.reduce_sum(
                tf.nn.log_poisson_loss(Y, Y_hat)
            )

        elif np.logical_and(self.loss_type == "poisson", self.activation != "exp"):
            self.loss_func = lambda Y, Y_hat, Y_act: tf.reduce_sum(
                Y_act - Y * tf.log(Y_act + 1e-33)
            )

        elif self.loss_type == "gaussian":
            self.loss_func = lambda Y, Y_hat, Y_act: tf.reduce_sum(tf.square(Y - Y_act))

        elif np.logical_and(
            self.loss_type == "binominal", self.activation == "sigmoid"
        ):
            # use pre-activation value with tf.nn.sigmoid_cross_entropy_with_logits
            self.loss_func = lambda Y, Y_hat, Y_act: tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_hat)
            )

    def forward(self, X, w, w0):
        """<Function> GLM forward model
        Input parameters::
        X: design matrix, tensor or ndarray of shape (n_samples, n_features)
        w: weight matrix, tensor or ndarray of shape (n_features, n_responses)
        w0: intercept matrix, tensor or ndarray of shape (1, n_responses)

        Returns::
        Y_act: model output (after activation)
        Y_hat: pre-activation output
        """

        Y_hat = tf.matmul(X, w) + w0
        Y_act = self.act_func(Y_hat)
        return Y_act, Y_hat

    def fit(
        self,
        X,
        Y,
        initial_w0=None,
        initial_w=None,
        feature_group_size=None,
        verbose=True,
    ):
        """
        <Method> Fit GLM
        Input parameters::
        X: design matrix, ndarray of shape (n_samples, n_features)
        Y: response matrix, ndarray of shape (n_samples, n_responses)
        initial_w0: optional, initial values of intercepts, ndarray of shape (n_responses,)
        initial_w: optional, initial values of weights, ndarray of shape (n_features, n_responses)
        feature_group_size: size of each group for regularization = 'group_lasso' or smooth_strength > 0.,
                            list of positive integer of len = n_groups;
                            the sum of all elements in this list must be equal to n_features,
                            and the features in X (axis 1) have to be sorted in corresponding orders,
                            as all features in group 0, followed by all features in group 1, all features in group 2, ..., etc.
        verbose: print loss during fitting or not, bool, default = True

        Returns::
        self
        """

        # check number of samples in X and Y
        assert (
            X.shape[0] == Y.shape[0]
        ), "Error: Number of samples (axis 0) of X and Y not matching!"

        # reshape X and Y if there's only one dimension
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # get dimension
        self.n_responses = Y.shape[1]
        self.n_features = X.shape[1]

        # convert inputs to tensor
        Xt = tf.convert_to_tensor(X, dtype=tf.float32)
        Yt = tf.convert_to_tensor(Y, dtype=tf.float32)

        # generate group matrix from feature group size if regularization = group_lasso
        if self.regularization == "group_lasso":
            assert (
                feature_group_size is not None
            ), "Error: You must provide group_size_list for group_lasso regularization!"
            assert (
                np.sum(np.array(feature_group_size)) == self.n_features
            ), "Error: Sum of group_size_list is not equal to number of features (X.shape[1])!"
            group_matrix = make_group_matrix(feature_group_size)
            self.group_matrix = tf.convert_to_tensor(group_matrix, dtype=tf.float32)
            self.feature_group_size = tf.convert_to_tensor(
                feature_group_size, dtype=tf.float32
            )

        # generate prior matrix from feature group size if smoothness penalty is non-zero
        if self.smooth_strength > 0.0:
            assert (
                feature_group_size is not None
            ), "Error: You must provide group_size_list for smooth_strength > 0"
            assert (
                np.sum(np.array(feature_group_size)) == self.n_features
            ), "Error: Sum of group_size_list is not equal to number of features (X.shape[1])!"
            prior_matrix = make_prior_matrix(feature_group_size)
            self.prior_matrix = tf.convert_to_tensor(prior_matrix, dtype=tf.float32)

        # find initial values of w0 and w
        if initial_w0 is not None:
            initial_w0 = initial_w0.reshape(1, -1)
            assert (
                initial_w0.shape[1] == self.n_responses
            ), "Error: Incorrect shape of initial_w0!"
        else:
            initial_w0 = tf.random.normal(
                [1, self.n_responses], mean=1e-5, stddev=1e-5, dtype=tf.float32
            )

        if initial_w is not None:
            assert (
                initial_w.shape[0] == self.n_features
                and initial_w.shape[1] == self.n_responses
            ), "Error: Incorrect shape of initial_w!"
        else:
            initial_w = tf.random.normal(
                [self.n_features, self.n_responses],
                mean=1e-5,
                stddev=1e-5,
                dtype=tf.float32,
            )

        # initialize variables
        w0 = tf.Variable(initial_w0, trainable=True, name="intercept", dtype=tf.float32)
        w = tf.Variable(initial_w, trainable=True, name="weight", dtype=tf.float32)

        # compute average null deviance
        null_dev = np.full((self.n_responses,), np.NaN)
        for ii in range(self.n_responses):
            this_Y = Yt[:, ii]
            null_dev[ii] = null_deviance(this_Y, loss_type=self.loss_type)

        avg_dev = np.sum(null_dev) / Y.shape[0] / self.n_responses

        # fit the model
        start_time = time.time()
        w_series, loss_trace, lambda_trace = self._fit(
            Xt, Yt, w, w0, avg_dev, verbose=verbose
        )
        if verbose:
            print("Fitting took {:1.2f} seconds.".format(time.time() - start_time))

        self.w_series = w_series
        self.loss_trace = loss_trace
        self.lambda_trace = lambda_trace

        self.fitted = True

    def _compute_loss(self, Xt, Yt, w, w0, lambda_index):
        """
        <Function> Compute loss with regularization, used in <Function>_fit
        Input parameters::
        Xt: design matrix, tensor of shape (n_samples, n_features)
        Yt: response matrix, tensor of shape (n_samples, n_responses)
        w: intercept matrix, tensor of shape (1, n_responses)
        w0: weight matrix, tensor of shape (n_features, n_responses)
        lambda_index: index for lambda for regularization

        Returns::
        loss: average loss with regularization
        """
        # get number of timepoints
        n_t = Xt.shape[0]

        # pass through forward model
        Y_act, Y_hat = self.forward(Xt, w, w0)

        # compute loss
        loss = self.loss_func(Yt, Y_hat, Y_act) / n_t / self.n_responses

        # add regularization to loss
        if self.regularization == "elastic_net":
            loss += (
                self.lambda_series[lambda_index] * self.reg_func(w) / self.n_responses
            )
        elif self.regularization == "group_lasso":
            loss += (
                self.lambda_series[lambda_index]
                * self.reg_func(w, self.group_matrix, self.feature_group_size)
                / self.n_responses
            )
        if self.smooth_strength > 0.0:
            loss += self.smooth_reg_func(w, self.prior_matrix) / self.n_responses

        return loss

    def _fit(self, Xt, Yt, w, w0, avg_dev, prev_w_series=None, verbose=True):
        """
        <Function> Fit the model with gradient descent, used in <Method> fit
        Input parameters::
        Xt: design matrix, tensor of shape (n_samples, n_features)
        Yt: response matrix, tensor of shape (n_samples, n_responses)
        w: intercept matrix, tensor of shape (1, n_responses)
        w0: weight matrix, tensor of shape (n_features, n_responses)
        avg_dev: average null deviance per sample per response
        prev_w_series: optional, w_series in previous fit (or some initial values),
                       list of len n_lambdas as [[w0, w] for lambda 1, [w0, w] for lambda 2, ..., etc.]
        verbose: print loss during fitting or not, bool

        Returns::
        w_series: fitted intercepts and weights for all lambdas,
                  list of len n_lambdas as [[w0, w] for lambda 1, [w0, w] for lambda 2, ..., etc.]
        loss_trace: list of training loss on each iteration during fitting
        lambda_trace: list of lambda value for each iteration during fitting
        """
        # select optimizer
        if self.optimizer == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == "sgdm":
            opt = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate, momentum=self.momentum
            )

        # prelocate
        w_series = []
        loss_trace = []
        lambda_trace = []

        # fit model
        lambda_index = 0
        iter_this_lambda = 0

        # re-initialize weights if prev_w_series is given
        if prev_w_series is not None:
            # weights from prev_w_series
            prev_w0 = prev_w_series[lambda_index][0]
            prev_w = prev_w_series[lambda_index][1]

            # random weights
            random_w0 = tf.random.normal(
                [1, self.n_responses], mean=1e-5, stddev=1e-5, dtype=tf.float32
            )
            random_w = tf.random.normal(
                [self.n_features, self.n_responses],
                mean=1e-5,
                stddev=1e-5,
                dtype=tf.float32,
            )

            # compute loss for the new lambda with current weights, previous weights, and random weights
            loss_current = self._compute_loss(Xt, Yt, w, w0, lambda_index)
            loss_random = self._compute_loss(Xt, Yt, random_w, random_w0, lambda_index)
            loss_prev = self._compute_loss(Xt, Yt, prev_w, prev_w0, lambda_index)

            # reset the weights with the lowest loss
            if loss_current > loss_random and loss_current > loss_prev:
                if loss_prev < loss_random:
                    w0.assign(prev_w0)
                    w.assign(prev_w)
                else:
                    w0.assign(random_w0)
                    w.assign(random_w)

        while True:
            iter_this_lambda += 1

            # compute loss
            with tf.GradientTape() as tape:
                loss_this_iter = self._compute_loss(Xt, Yt, w, w0, lambda_index)

            loss_this_iter_num = loss_this_iter.numpy()
            assert not np.isnan(loss_this_iter_num), "Loss is nan -- check."
            loss_trace.append(loss_this_iter_num)
            lambda_trace.append(self.lambda_series[lambda_index])

            # compute, process and apply gradient
            grads = tape.gradient(loss_this_iter, [w0, w])
            processed_grads = [
                tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) for g in grads
            ]
            opt.apply_gradients(zip(processed_grads, [w0, w]))

            # Check for convergence:
            if (iter_this_lambda % self.num_iter_check) == 0 and (
                iter_this_lambda >= self.min_iter_per_lambda
            ):
                loss_diff = np.mean(
                    -np.diff(loss_trace[-self.num_iter_check :])
                )  # Mean over last X steps.

                if (np.absolute(loss_diff) < self.convergence_tol * avg_dev) or (
                    iter_this_lambda >= self.max_iter_per_lambda
                ):
                    if verbose:
                        if iter_this_lambda >= self.max_iter_per_lambda:
                            print(
                                "Fitting with Lambda {} iter {} did not converge (loss diff = {:1.8f})".format(
                                    lambda_index, iter_this_lambda, loss_diff
                                )
                            )
                        else:
                            print(
                                "Fitting with Lambda {} iter {} converged (loss diff = {:1.8f})".format(
                                    lambda_index, iter_this_lambda, loss_diff
                                )
                            )

                    # collect current w and w0 if converges for this lambda
                    w_series.append([w0.numpy(), w.numpy()])

                    lambda_index += 1
                    iter_this_lambda = 0

                    if lambda_index < self.lambda_series.shape[0]:

                        # re-initialize weights for next lambda if prev_w_series is given
                        if prev_w_series is not None:
                            # weights from prev_w_series
                            prev_w0 = prev_w_series[lambda_index][0]
                            prev_w = prev_w_series[lambda_index][1]

                            # random weights
                            random_w0 = tf.random.normal(
                                [1, self.n_responses],
                                mean=1e-5,
                                stddev=1e-5,
                                dtype=tf.float32,
                            )
                            random_w = tf.random.normal(
                                [self.n_features, self.n_responses],
                                mean=1e-5,
                                stddev=1e-5,
                                dtype=tf.float32,
                            )

                            # compute loss for the new lambda with current weights, previous weights, and random weights
                            loss_current = self._compute_loss(
                                Xt, Yt, w, w0, lambda_index
                            )
                            loss_random = self._compute_loss(
                                Xt, Yt, random_w, random_w0, lambda_index
                            )
                            loss_prev = self._compute_loss(
                                Xt, Yt, prev_w, prev_w0, lambda_index
                            )

                            # reset the weights with the lowest loss
                            if loss_current > loss_random and loss_current > loss_prev:
                                if loss_prev < loss_random:
                                    w0.assign(prev_w0)
                                    w.assign(prev_w)
                                else:
                                    w0.assign(random_w0)
                                    w.assign(random_w)

                        # reset optimizer
                        if opt == "adam":
                            optimizer = tf.keras.optimizers.Adam(
                                learning_rate=self.learning_rate
                            )
                        elif opt == "sgdm":
                            optimizer = tf.keras.optimizers.SGD(
                                learning_rate=self.learning_rate, momentum=self.momentum
                            )

                    elif lambda_index == self.lambda_series.shape[0]:
                        if verbose:
                            print("Finished lambda series.")
                        break
                else:
                    if verbose:
                        print(
                            "Lambda {} iter {} loss: {:1.8f} diff: {:1.8f}".format(
                                lambda_index,
                                iter_this_lambda,
                                loss_this_iter,
                                loss_diff,
                            )
                        )

        return w_series, loss_trace, lambda_trace

    def _calculate_fit_quality(self, X, Y):
        """
        <Function> Make prediction and calculate fit quality (fraction deviance explained) for all lambda values,
                   used in <Method> select_model
        Input parameters::
        X: design matrix, tensor or ndarray of shape (n_samples, n_features)
        Y: response matrix, tensor or ndarray of shape (n_samples, n_responses)

        Returns::
        all_frac_dev_expl: fraction explained deviance for all responses, ndarray of shape (n_lambdas, n_response)
        all_d_model: model deviance for all responses, ndarray of shape (n_lambdas, n_response)
        all_d_null: null deviance for all responses, ndarray of shape (n_lambdas, n_response)
        all_prediction: prediction for Y for all lambdas, list of len n_lambdas
        """

        # prelocate
        all_frac_dev_expl = []
        all_d_model = []
        all_d_null = []
        all_prediction = []

        for idx, w in enumerate(self.w_series):
            # make predictions
            prediction, _ = self.forward(X, w[1], w[0])
            all_prediction.append(prediction.numpy())
            # calculate fraction deviance explained, model deviance, and null deviance
            frac_dev_expl, d_model, d_null = deviance(
                prediction, Y, loss_type=self.loss_type
            )
            all_d_model.append(d_model)
            all_frac_dev_expl.append(frac_dev_expl)
            if idx == 0:
                all_d_null = d_null
        all_frac_dev_expl = np.stack(all_frac_dev_expl, axis=0)
        all_d_model = np.stack(all_d_model, axis=0)

        return all_frac_dev_expl, all_d_model, all_d_null, all_prediction

    def select_model(self, X_val, Y_val, min_lambda=0.0, make_fig=True):
        """
        <Method> Select model with the highest fraction deviance explained using validation data.
                 Must be called after fitting.
        Input parameters::
        X_val: design matrix for validation, tensor or ndarray of shape (n_samples, n_features)
        Y_val: response matrix for validation, tensor or ndarray of shape (n_samples, n_responses)
        min_lambda: value of minimal lambda for selection, float
        make_fig: generate plots or not, bool

        Returns::
        self
        """

        # reshape X_val and Y_val if there's only one dimension
        if X_val.ndim == 1:
            X_val = X_val.reshape(-1, 1)
        if Y_val.ndim == 1:
            Y_val = Y_val.reshape(-1, 1)

        # sanity check
        assert self.fitted, "Error: You have not fitted the model!"
        assert (
            X_val.shape[0] == Y_val.shape[0]
        ), "Error: Number of datapoints (axis 0) of X_val and Y_val not matching!"
        assert (
            X_val.shape[1] == self.n_features
        ), "Error: Incorrect number of features (axis 1) in X_val!"
        assert (
            Y_val.shape[1] == self.n_responses
        ), "Error: Incorrect number of repsonse (axis 1) in Y_val!"

        # calculate fit quality (frac deviance explained) using validation set
        all_frac_dev_expl, _, _, _ = self._calculate_fit_quality(X_val, Y_val)

        # select best model for each source
        if min_lambda > self.lambda_series.min():
            min_lambda_idx = np.argwhere(self.lambda_series < min_lambda)[0][0] - 1
            self.selected_lambda_ind = np.minimum(
                np.argmax(all_frac_dev_expl, axis=0), min_lambda_idx
            )
        else:
            self.selected_lambda_ind = np.argmax(all_frac_dev_expl, axis=0)

        self.selected_lambda = self.lambda_series[self.selected_lambda_ind]

        selected_w0 = []
        selected_w = []
        selected_frac_dev_expl_val = []
        for idx in range(self.n_responses):
            best_lambda_ind = self.selected_lambda_ind[idx]
            best_w0 = self.w_series[best_lambda_ind][0][:, idx]
            best_w = self.w_series[best_lambda_ind][1][:, idx]
            selected_w0.append(best_w0)
            selected_w.append(best_w)
            selected_frac_dev_expl_val.append(all_frac_dev_expl[best_lambda_ind, idx])

        self.selected_w0 = np.stack(selected_w0, axis=0).reshape(
            -1,
        )
        self.selected_w = np.stack(selected_w, axis=1)
        self.selected_frac_dev_expl_val = np.stack(selected_frac_dev_expl_val, axis=0)
        self.selected = True

        if make_fig:
            self._model_selection_plot(all_frac_dev_expl, selected_frac_dev_expl_val)

    def _model_selection_plot(
        self, all_frac_dev_expl, selected_frac_dev_expl, bin_width=0.05
    ):
        """
        <Function> Make plots for model selection, used in <Method> select_model
        Input parameters::
        all_frac_dev_expl: fraction explained deviance for all responses, ndarray of shape (n_lambdas, n_response)
        selected_frac_dev_expl: fraction deviance explained for selected models

        Returns::
        self
        """

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].plot(
            np.log10(self.lambda_series), all_frac_dev_expl, color="k", linewidth=0.5
        )
        axes[0].set_ylim((-0.1, 1))
        axes[0].set_xlabel("log_lambda")
        axes[0].set_ylabel("Fraction deviance explained")
        axes[0].set_title("Fraction deviance explained vs. lambda")

        axes[1].hist(selected_frac_dev_expl, bins=np.arange(0, 1, bin_width))
        axes[1].set_xlabel("Fraction deviance explained")
        axes[1].set_ylabel("Count")
        axes[1].set_title(
            "Distribution of fraction deviance explained \n for selected models on validation data"
        )
        plt.tight_layout()

    def predict(self, X):
        """<Method> Make prediction using selected model weights. Must be called after model selection.
        Input parameters::
        X: design matrix for test, tensor or ndarray of shape (n_samples, n_features)

        Returns::
        Y_pred: predicted response matrix, ndarray of shape (n_samples, n_responses)
        """
        assert (
            self.selected
        ), "Error: You have to perform model selection with validation data first before making prediction!"

        # reshape X if there's only one dimension
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        assert (
            X.shape[1] == self.n_features
        ), "Error: Incorrect number of features (axis 1) in X!"

        # make prediction with forward model
        Y_act, _ = self.forward(X, self.selected_w, self.selected_w0.reshape(1, -1))
        Y_pred = Y_act.numpy()
        return Y_pred

    def evaluate(self, X_test, Y_test, make_fig=True):
        """<Method> Evaluate selected model with test data using selected weights.
                    Must be called after model selection.
        Input parameters::
        X_test: design matrix for test, tensor or ndarray of shape (n_samples, n_features)
        Y_test: response matrix for test, tensor or ndarray of shape (n_samples, n_responses)
        make_fig: generate plots or not, bool

        Returns::
        frac_dev_expl: fraction deviance explained for all responses, ndarray of shape (n_responses,)
        dev_model: model deviance for all responses, ndarray of shape (n_responses,)
        dev_null: null deviance for all responses, ndarray of shape (n_responses,)
        dev_expl: deviance explained for all responses (null deviance - model deviance), ndarray of shape (n_responses,)
        """

        # reshape X_test and Y_test if there's only one dimension
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        if Y_test.ndim == 1:
            Y_test = Y_test.reshape(-1, 1)

        # sanity check
        assert (
            self.selected
        ), "Error: You have to perform model selection with validation data first before evaluating!"
        assert (
            X_test.shape[0] == Y_test.shape[0]
        ), "Error: Number of datapoints (axis 0) of X_test and Y_test not matching!"
        assert (
            X_test.shape[1] == self.n_features
        ), "Error: Incorrect number of features (axis 1) in X_test!"
        assert (
            Y_test.shape[1] == self.n_responses
        ), "Error: Incorrect number of repsonse (axis 1) in Y_test!"

        # select best model for each source
        frac_dev_expl = []
        dev_model = []
        dev_null = []
        dev_expl = []

        # make prediction on test set and calculate fraction deviance explained, model deviance, and null deviance
        prediction = self.predict(X_test)
        for idx in range(self.n_responses):
            best_frac_deviance, best_d_model, best_d_null = deviance(
                prediction[:, idx], Y_test[:, idx], loss_type=self.loss_type
            )
            best_dev_expl = best_d_null - best_d_model

            frac_dev_expl.append(best_frac_deviance)
            dev_model.append(best_d_model)
            dev_null.append(best_d_null)
            dev_expl.append(best_dev_expl)

        print(
            "Fraction deviance explained: mean = {:1.4f}, median = {:1.4f}".format(
                np.mean(np.array(frac_dev_expl)), np.median(np.array(frac_dev_expl))
            )
        )

        if make_fig:
            # plot CDF for fraction deviance explained / scatter of  deviance explained vs null deviance
            density, bins = np.histogram(
                frac_dev_expl, bins=np.arange(0, 1, 0.01), density=True
            )
            this_ecdf = np.cumsum(density)

            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].plot(bins[1:], this_ecdf)
            axes[0].set_xlabel("Fraction deviance explained")
            axes[0].set_ylabel("Cumulative density")
            axes[0].set_title("CDF for fraction deviance explained")

            axes[1].plot(dev_null, dev_expl, ".", markersize=3)
            axes[1].plot(
                np.linspace(0, np.max(dev_null), 100),
                np.linspace(0, np.max(dev_null), 100),
                linestyle="--",
                linewidth=1,
                color=(0.5, 0.5, 0.5),
            )
            axes[1].set_xlim([0, np.max(dev_null)])
            axes[1].set_ylim([0, np.max(dev_expl)])
            axes[1].set_xlabel("Deviance for null model")
            axes[1].set_ylabel("Deviance explained")
            axes[1].set_title("Deviance explained vs. null deviance")
            plt.tight_layout()

        frac_dev_expl = np.array(frac_dev_expl)
        dev_model = np.array(dev_model)
        dev_null = np.array(dev_null)
        dev_expl = np.array(dev_expl)

        return frac_dev_expl, dev_model, dev_null, dev_expl


#### GLM_CV class ####


class GLM_CV(GLM):
    def __init__(
        self,
        n_folds=5,
        auto_split=True,
        split_by_group=True,
        split_random_state=None,
        activation="exp",
        loss_type="poisson",
        regularization="elastic_net",
        lambda_series=10.0 ** np.linspace(-1, -8, 30),
        l1_ratio=0.0,
        smooth_strength=0.0,
        optimizer="adam",
        learning_rate=1e-3,
        momentum=0.5,
        min_iter_per_lambda=100,
        max_iter_per_lambda=10**4,
        num_iter_check=100,
        convergence_tol=1e-6,
    ):
        """
        GLM_CV class, inherited from GLM class
        It fits GLM with n fold cross validation,
        selects proper regularization values for each response based on deviance of CV held-out data,
        and returns weights and intercepts from models re-fitted with all datapoints in training data with selected regularization values.
        Fit Y with multiple responses simultaneously.

        Model: Y = activation(X * w + w0) + noise
        X: n_samples x n_features
        Y: n_samples x n_responses
        w: n_features x n_responses
        w0: n_responses

        Use following combinations of activation and loss_type for common model types:
        Gaussian: 'linear' + 'gaussian'; Poisson: 'exp' + 'poisson'; Logistic: 'sigmoid' + 'binominal'
        Or create you own combinations

        Input parameters::
        (Unique to GLM_CV class)
        n_folds: number of CV folds, default = 5
        auto_split: perform CV split automatically or not, bool, default = True
        split_by_group: perform CV split according to a third-party provided group when auto_split = True, default = True
        split_random_state: optional, numpy random state for CV splitting

        (Same as GLM class)
        activation: {'linear', 'exp', 'sigmoid', 'relu', 'softplus'}, default = 'exp'
        loss_type: {'gaussian', 'poisson', 'binominal'}, default = 'poisson'
        regluarization: {'elastic_net', 'group_lasso'}, default = 'elastic_net'
        lambda_series: list or ndarray of a series of regularization strength (lambda), in descending order,
                       default = 10.0 ** np.linspace(-1, -8, 30)
        l1_ratio: L1 ratio for elastic_net regularization (l1_ratio = 1. is Lasso, l1_ratio = 0. is ridge), default = 0.
        smooth_strength: strength for smoothness penalty, default = 0.
        optimizer: {'adam', 'sgdm'}, default = 'adam'
        learning_rate: learning rate for optimizer, default = 1e-3
        momentum: momentum for sgdm optimizer, default = 0.5
        min_iter_per_lambda: minimal iterations for each lambda, default = 100
        max_iter_per_lambda: maximal iterations for each lambda, default = 10000
        num_iter_check: number of iterations for checking convergence (delta loss is averaged over this number), default = 100
        convergence_tol: convergence criterion, complete fitting when absolute average delta loss over past num_iter_check iterations
                         is smaller than convergence_tol*average null deviance of the data

        Attributes::
        Key attributes you might look up after fitting and model selection:
          selected_w0, selected_w, selected_lambda, selected_lambda_ind, selected_frac_dev_expl_cv

        List of all attributes:
        (Same as GLM class)
        act_func: activation function based on input activation type (tensorflow function)
        loss_func: loss function based on input loss type (tensorflow function)
        reg_func: regularization function based on input regularization type (tensorflow function)
        fitted: if the model has been fitted, bool
        selected: if model selection has been performed, bool
        n_features: number of features seen during fit (X.shape[1])
        n_responses: number of responses seen during fit (Y.shape[1])
        n_lambdas: number of regularization strengths (lambda_series.shape[0])
        feature_group_size: size of each group for regularization = 'group_lasso', list of len = n_groups,
                            provided by the user as input to fit method when regularization = 'group_lasso' or smooth_strength > 0.
        group_matrix: group matrix used in fitting for regularization = 'group_lasso', a matrix that converts from a n_group vector
                      to an n_expanded_group vector for scaling the groups differently, tensor of shape (n_groups, n_features)
        prior_matrix: prior matrix used in fitting for smooth_strength > 0., a block-diagonal matrix containing [-1, 2, 1] on the
                      diagonal for expanded features, tensor of shape (n_features, n_features)
        selected_w0: intercepts for selected models (from the re-fitted models using all datapoints), ndarray of shape (n_responses,)
        selected_w: weights for selected models (from the re-fitted models using all datapoints),
                    ndarray of shape (n_features, n_responses)
        selected_lambda: lambda values for each response for selected models, ndarray of shape (n_responses,)
        selected_lambda_ind: indices of lambdas for each response for selected models, ndarray of shape (n_responses,)

        (Unique to GLM_CV class)
        w_series_dict: all fitted intercepts and weights for all lambdas across all folds,
                       dictionary arranged as {n_fold: [[w0, w] for lambda 1, [w0, w] for lambda 2,...]} for fold 0 to n_folds-1,
                       additionally, w_series_dict[n_folds] is the w_series when fitting with all datapoints
        selected_frac_dev_expl_cv: fraction deviance explained evaluated on cv held-out data for selected models
        split_random_state: numpy random state for CV splitting, either given by user or generated automatically
        train_idx: indices for training data for each fold, either given by user or generated through auto_split,
                   dictionary arranges as {n_fold: ndarray containing training indices of that fold for n_fold in range(n_folds)},
                   additionally, train_idx[n_folds] returns the indices for all datapoints which is used for a final fit
        val_idx: indices for CV held-out data for each fold, either given by user or generated through auto_split,
                 dictionary arranges as {n_fold: ndarray containing validation indices of that fold for n_fold in range(n_folds)}
        Y_fit: values of response matrix used in fitting; used in model selection
        all_prediction: prediction made on CV held-out data for each fold during fitting; used in model selection
        all_deviance: model deviance computed on CV held-out data for each fold during fitting; used in model selection

        Methods::
        fit(X, Y, [initial_w0, initial_w, feature_group_size, verbose]):
            fit GLM to training data with cross validation
        select_model([se_fraction, min_lambda, make_fig]): select model after fit is called
        predict(X): returns prediction on input data X using selected models after select_model is called; inherited from GLM class
        evaluate(X, Y, [make_fig]): compute fraction deviance explained on input data X, Y using selected models
                                    after select_model is called; inherited from GLM class
        """

        super().__init__(
            activation,
            loss_type,
            regularization,
            lambda_series,
            l1_ratio,
            smooth_strength,
            optimizer,
            learning_rate,
            momentum,
            min_iter_per_lambda,
            max_iter_per_lambda,
            num_iter_check,
            convergence_tol,
        )

        self.n_folds = n_folds
        self.auto_split = auto_split
        self.split_by_group = split_by_group

        # if no auto-split, change split_by_group to False
        if self.auto_split == False:
            self.split_by_group = False

        if split_random_state is not None:
            self.split_random_state = split_random_state
        else:
            self.split_random_state = np.random.randint(0, high=2**31)

    def _cv_split(self, X, Y, group_idx=None):
        """
        <Function> Split data into train-validation set for different CV folds, used by <Method> fit
        Input parameters::
        X: design matrix, ndarray of shape (n_samples, n_features)
        Y: response matrix, ndarray of shape (n_samples, n_responses)
        group_idx: third-party provided group for each sample for split_by_group = True, ndarray of shape (n_samples, )

        Returns::
        train_idx: indices for training data for each fold,
                   dictionary arranges as {n_fold: ndarray containing training indices for that fold} for fold 0 to n_folds-1,
                   additionally, train_idx[n_folds] returns the indices for all datapoints which is used for a final fit
        val_idx: indices for CV held-out data for each fold,
                 dictionary arranges as {n_fold: ndarray containing validation indices for that fold} for fold 0 to n_folds-1
        """

        # choose splitter
        if self.split_by_group:
            assert (
                group_idx is not None
            ), "Error: You must supply group_idx if split_by_group = True"
            assert (
                group_idx.shape[0] == Y.shape[0]
            ), "Error: Number of timepoints (axis 0) of group_idx and data not matching!"
            np.random.seed(self.split_random_state)
            splitter = GroupKFold(n_splits=self.n_folds).split(X, Y, group_idx)
        else:
            splitter = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.split_random_state,
            ).split(X, Y)

        # create dictionaries to hold train and validation idx for each fold
        train_idx = {}
        val_idx = {}

        # split data and save the train and validation index
        for n_fold, (train_index, val_index) in enumerate(splitter):
            train_idx[n_fold] = train_index
            val_idx[n_fold] = val_index

        # we fit the model to all datapoints at the end and report weights of this fit after model selection
        train_idx[self.n_folds] = np.arange(Y.shape[0])

        return train_idx, val_idx

    def fit(
        self,
        X,
        Y,
        train_idx=None,
        val_idx=None,
        group_idx=None,
        initial_w=None,
        initial_w0=None,
        feature_group_size=None,
        verbose=True,
    ):
        """
        <Method> Fit GLM_CV. This method overwrites the fit method in GLM class.
                 This method loops over each CV fold to perform fitting + one final round of fitting using all datapoints;
                 it also saves the prediction and model deviance on CV held-out data that are used in model selection process.
        Input parameters::
        X: design matrix, ndarray of shape (n_samples, n_features)
        Y: response matrix, ndarray of shape (n_samples, n_responses)
        train_idx: indices for training data for each fold if auto_split = False,
                   dictionary arranges as {n_fold: ndarray containing training indices of that fold for n_fold in range(n_folds)},
                   additionally, please include train_idx[n_folds] which returns the indices for all datapoints which is used
                   for a final fit
        val_idx: indices for CV held-out data for each fold if auto_split = False,
                 dictionary arranges as {n_fold: ndarray containing validation indices of that fold for n_fold in range(n_folds)}
        group_idx: third-party provided group for each sample for auto_split = True and split_by_group = True,
                   ndarray of shape (n_samples, )
        initial_w0: optional, initial values of intercepts, ndarray of shape (n_responses,)
        initial_w: optional, initial values of weights, ndarray of shape (n_features, n_responses)
        feature_group_size: size of each group for regularization = 'group_lasso' or smooth_strength > 0.,
                    list of positive integer of len = n_groups;
                    the sum of all elements in this list must be equal to n_features,
                    and the features in X (axis 1) have to be sorted in corresponding orders,
                    as all features in group 0, followed by all features in group 1, all features in group 2, ..., etc.
        verbose: print loss during fitting or not, bool

        Returns::
        self
        """
        # split data if needed
        if self.auto_split:
            train_idx, val_idx = self._cv_split(X, Y, group_idx=group_idx)
        # check train_idx and val_idx
        else:
            assert (
                train_idx is not None and val_idx is not None
            ), "Error: You must supply train_idx and val_idx if auto_split = False"
            for n_fold in range(self.n_folds):
                assert (
                    n_fold in train_idx.keys() and n_fold in val_idx.keys()
                ), "Error: Incorrect format of train_idx or val_idx. Check!"
            # add indices to all datapoint for final fit if not provided in train_idx
            if self.n_folds not in train_idx.keys():
                train_idx = {self.n_folds: np.arange(Y.shape[0])}

        self.train_idx = train_idx
        self.val_idx = val_idx

        ##### This part is identical to the first part of fit method for class GLM ######
        # check number of samples in X and Y
        assert (
            X.shape[0] == Y.shape[0]
        ), "Error: Number of samples (axis 0) of X and Y not matching!"

        # reshape Y if there's only one response
        # reshape X and Y if there's only one dimension
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # get dimension
        self.n_responses = Y.shape[1]
        self.n_features = X.shape[1]

        # convert inputs to tensor
        Xt = tf.convert_to_tensor(X, dtype=tf.float32)
        Yt = tf.convert_to_tensor(Y, dtype=tf.float32)

        # generate group matrix from feature group size if regularization = group_lasso
        if self.regularization == "group_lasso":
            assert (
                feature_group_size is not None
            ), "Error: You must provide group_size_list for group_lasso regularization!"
            assert (
                np.sum(np.array(feature_group_size)) == self.n_features
            ), "Error: Sum of group_size_list is not equal to number of features (X.shape[1])!"
            group_matrix = make_group_matrix(feature_group_size)
            self.group_matrix = tf.convert_to_tensor(group_matrix, dtype=tf.float32)
            self.feature_group_size = tf.convert_to_tensor(
                feature_group_size, dtype=tf.float32
            )

        # generate prior matrix from feature group size if smoothness penalty is non-zero
        if self.smooth_strength > 0.0:
            assert (
                feature_group_size is not None
            ), "Error: You must provide group_size_list for smooth_strength > 0"
            assert (
                np.sum(np.array(feature_group_size)) == self.n_features
            ), "Error: Sum of group_size_list is not equal to number of features (X.shape[1])!"
            prior_matrix = make_prior_matrix(feature_group_size)
            self.prior_matrix = tf.convert_to_tensor(prior_matrix, dtype=tf.float32)

        # find initial values of w0 and w
        if initial_w0 is not None:
            initial_w0 = initial_w0.reshape(1, -1)
            assert (
                initial_w0.shape[1] == self.n_responses
            ), "Error: Incorrect shape of initial_w0!"
        else:
            initial_w0 = tf.random.normal(
                [1, self.n_responses], mean=1e-5, stddev=1e-5, dtype=tf.float32
            )

        if initial_w is not None:
            assert (
                initial_w.shape[0] == self.n_features
                and initial_w.shape[1] == self.n_responses
            ), "Error: Incorrect shape of initial_w!"
        else:
            initial_w = tf.random.normal(
                [self.n_features, self.n_responses],
                mean=1e-5,
                stddev=1e-5,
                dtype=tf.float32,
            )

        # initialize variables
        w0 = tf.Variable(initial_w0, trainable=True, name="intercept", dtype=tf.float32)
        w = tf.Variable(initial_w, trainable=True, name="weight", dtype=tf.float32)

        # compute average null deviance
        null_dev = np.full((self.n_responses,), np.NaN)
        for ii in range(self.n_responses):
            this_Y = Yt[:, ii]
            null_dev[ii] = null_deviance(this_Y, loss_type=self.loss_type)

        avg_dev = np.sum(null_dev) / Y.shape[0] / self.n_responses
        ######

        # save target Y for fitting (used in model selection later)
        self.Y_fit = Y

        # prelocate
        self.w_series_dict = {}
        self.all_prediction = [
            np.full(Y.shape, np.NaN) for idx, _ in enumerate(self.lambda_series)
        ]
        self.all_deviance = [
            np.full((self.n_folds, self.n_responses), np.NaN)
            for idx, _ in enumerate(self.lambda_series)
        ]

        # fit the model
        start_time = time.time()
        for n_fold in range(
            self.n_folds + 1
        ):  # when n_fold == self.n_folds, the fit is using all datapoints
            print("n_fold =", n_fold)
            start_time_fold = time.time()
            X_train = tf.gather(Xt, train_idx[n_fold])
            Y_train = tf.gather(Yt, train_idx[n_fold])

            if n_fold == 0:
                w_series, _, _ = self._fit(
                    X_train, Y_train, w, w0, avg_dev, verbose=verbose
                )
            else:
                # provide prev_w_series for better initialization with faster fitting speed
                w_series, _, _ = self._fit(
                    X_train,
                    Y_train,
                    w,
                    w0,
                    avg_dev,
                    prev_w_series=self.w_series_dict[n_fold - 1],
                    verbose=verbose,
                )

            self.w_series_dict[n_fold] = w_series
            if verbose:
                print(
                    "Fitting for this fold took {:1.2f} seconds.".format(
                        time.time() - start_time_fold
                    )
                )

            # make prediction and compute deviance on CV held-out data for the current fold (used in model selection later)
            if n_fold < self.n_folds:
                these_val_idx = val_idx[n_fold]
                X_val = X[these_val_idx, :]
                Y_val = Y[these_val_idx, :]
                for this_lambda_idx, this_w in enumerate(w_series):
                    prediction, _ = self.forward(X_val, this_w[1], this_w[0])
                    _, d_model, _ = deviance(
                        prediction.numpy(), Y_val, loss_type=self.loss_type
                    )
                    # save prediction for CV held-out data of this fold for computing and reporting fraction deviance explained
                    self.all_prediction[this_lambda_idx][
                        these_val_idx, :
                    ] = prediction.numpy()
                    # save deviance for CV held-out data of this fold for model selection
                    self.all_deviance[this_lambda_idx][n_fold, :] = d_model.reshape(
                        1, -1
                    )

        if verbose:
            print("Fitting took {:1.2f} seconds.".format(time.time() - start_time))

        self.fitted = True

    def _calculate_fit_quality_cv(self):
        """
        <Function> Calculate fit quality (fraction deviance explained) based on the prediction made on CV held-out data during fitting;
                   used in <Method> select_model.
                   Must be called after fitting.

        Returns::
        all_frac_dev_expl: fraction explained deviance for all responses, ndarray of shape (n_lambdas, n_response)
        all_d_model: model deviance for all responses, ndarray of shape (n_lambdas, n_response)
        all_d_null: null deviance for all responses, ndarray of shape (n_lambdas, n_response)
        """
        all_frac_dev_expl = []
        all_d_model = []

        for idx, _ in enumerate(self.lambda_series):
            prediction = self.all_prediction[idx]
            frac_dev_expl, d_model, d_null = deviance(
                prediction, self.Y_fit, loss_type=self.loss_type
            )
            all_d_model.append(d_model)
            all_frac_dev_expl.append(frac_dev_expl)
            if idx == 0:
                all_d_null = d_null
        all_frac_dev_expl = np.stack(all_frac_dev_expl, axis=0)
        all_d_model = np.stack(all_d_model, axis=0)

        return all_frac_dev_expl, all_d_model, all_d_null

    def select_model(self, se_fraction=1.0, min_lambda=0.0, make_fig=True):
        """
        <Method> Select models using prediction and model deviance computed on CV held-out data during fitting,
                 with se_fraction that controls the tolerance of choosing models with smallest deviance vs. larger regularization.
                 After selecting the proper regularization, the attributes selected_w0 and selected_w get assigned to the
                 intercepts and weights from the re-fitted models using all datapoints with the selected regularization values.
                 This method overwrites the select_model method in GLM class. Must be called after fitting.

        Input parameters::
        X_val: design matrix for validation, tensor or ndarray of shape (n_samples, n_features)
        Y_val: response matrix for validation, tensor or ndarray of shape (n_samples, n_responses)
        se_fraction: the fraction of standard error parametrizing the tolerance of choosing models
                     with smallest deviance vs. larger regularization
                     se_fraction = 0. means selecting models with the smallest model deviance (i.e. highest explained deviance);
                     se_fraction = 1. means selecting models using the "1SE rule";
                     or set se_fraction to arbitrary positive number to control the tolerance
        min_lambda: value of minimal lambda for selection, float
        make_fig: generate plots or not, bool

        Returns::
        self
        """

        # Sanity check
        assert self.fitted, "Error: You have not fitted the model!"

        # computer fraction deviance explained for all lambdas based on the prediction on CV held-out data
        all_frac_dev_expl_cv, _, _ = self._calculate_fit_quality_cv()

        # compute average deviance and standard error
        avg_deviance = [np.mean(dev, axis=0) for dev in self.all_deviance]
        avg_deviance = np.stack(avg_deviance, axis=0)
        se_deviance = [
            np.std(dev, axis=0) / np.sqrt(self.n_folds) for dev in self.all_deviance
        ]
        se_deviance = np.stack(se_deviance, axis=0)

        # prelocate
        selected_w0 = []
        selected_w = []
        selected_lambda = []
        selected_lambda_ind = []
        selected_frac_dev_expl_cv = []
        all_min_lambda_ind = []
        all_min_lambda = []

        # find minimal lambda index
        if min_lambda > self.lambda_series.min():
            min_lambda_idx = np.argwhere(self.lambda_series < min_lambda)[0][0] - 1
        else:
            min_lambda_idx = self.lambda_series.shape[0]

        for idx in range(self.n_responses):
            min_deviance = np.min(avg_deviance[:, idx])
            min_dev_lambda_ind = np.argmin(avg_deviance[:, idx])
            this_se = se_deviance[min_dev_lambda_ind, idx]
            threshold = min_deviance + this_se * se_fraction

            # find the lambda index with avg deviance smaller than threshold
            this_lambda_ind = np.argwhere(avg_deviance[:, idx] <= threshold)[0][0]
            this_lambda_ind = np.min([this_lambda_ind, min_lambda_idx])
            this_lambda = self.lambda_series[this_lambda_ind]

            # find fraction deviance explained for the selected lambda
            this_frac_dev = all_frac_dev_expl_cv[this_lambda_ind, idx]

            # find the corresponding weights for the lambda
            # note that w_series_dict[n_folds] returns the weights fitted with full data
            this_w0 = self.w_series_dict[self.n_folds][this_lambda_ind][0][:, idx]
            this_w = self.w_series_dict[self.n_folds][this_lambda_ind][1][:, idx]

            # collect all parameters
            selected_lambda_ind.append(this_lambda_ind)
            selected_lambda.append(this_lambda)
            all_min_lambda_ind.append(min_dev_lambda_ind)
            all_min_lambda.append(self.lambda_series[min_dev_lambda_ind])
            selected_w0.append(this_w0)
            selected_w.append(this_w)
            selected_frac_dev_expl_cv.append(this_frac_dev)

        self.selected_lambda_ind = np.array(selected_lambda_ind)
        self.selected_lambda = np.array(selected_lambda)
        self.min_lambda_ind = np.array(all_min_lambda_ind)
        self.min_lambda = np.array(all_min_lambda)
        self.selected_w0 = np.stack(selected_w0, axis=0)
        self.selected_w = np.stack(selected_w, axis=1)
        self.selected_frac_dev_expl_cv = np.stack(selected_frac_dev_expl_cv, axis=0)
        self.selected = True

        if make_fig:
            self._model_selection_plot(all_frac_dev_expl_cv, selected_frac_dev_expl_cv)

    def make_prediction_cv(self, X_ablated):
        """
        <Method> Make prediction on X_ablated using CV fold-specific weights of selected lambdas.
                 X_ablated must be identical to X_train (same samples) with certain features set to 0 or shuffle.
                 The prediction is made on the CV held-out data for each fold using the weights from that fold with selected lambdas.

        Input parameters::
        X_ablated: ablated design matrix identical to X_train (same samples) with certain features set to 0 or shuffle,
                   ndarray of shape (n_samples_train, n_features)

        Returns::
        pred: prediction made on X_ablated using CV fold-specific weights of selected lambdas,
              ndarray of shape (n_samples_train, n_responses)
        """

        # sanity check
        assert self.selected, "Error: You have not selected the model!"
        assert (
            X_ablated.shape[0] == self.Y_fit.shape[0]
        ), "Error: Number of samples (axis 0) in X_ablated is not identical to that of X_train during fitting!"

        # prelocate
        pred = np.empty((X_ablated.shape[0], self.n_responses))

        # loop over CV folds for making prediction on validation data
        for n_fold in range(self.n_folds):
            # grab validation indices for this fold
            these_val_frames = self.val_idx[n_fold]

            # grab w and w0 for this fold
            this_w0_cv = []
            this_w_cv = []
            for n_response in range(self.n_responses):
                this_lambda_ind = self.selected_lambda_ind[n_response]
                w0 = self.w_series_dict[n_fold][this_lambda_ind][0][:, n_response]
                w = self.w_series_dict[n_fold][this_lambda_ind][1][:, n_response]
                this_w0_cv.append(w0)
                this_w_cv.append(w)

            this_w_cv = np.stack(this_w_cv, axis=1)
            this_w0_cv = np.stack(this_w0_cv, axis=1)

            # make predictions on validation frames of this fold
            this_pred, _ = self.forward(
                X_ablated[these_val_frames, :], this_w_cv, this_w0_cv.reshape(1, -1)
            )
            pred[these_val_frames, :] = this_pred.numpy()

        return pred


#### Utility functions ####


def stable(x, eps=1e-33):
    """
    Add a tiny positive constant to input value to stablize it when taking log (avoid log(0))
    """
    return x + eps


def pointwise_deviance(y_true, y_pred, loss_type="poisson"):
    """
    Compute pointwise deviance for data with given loss type
    Input parameters::
    y_true: true values, ndarray
    y_pred: predicted values, ndarray
    loss_type: {'gaussian', 'poisson', 'binominal'}, default = 'poisson'

    Returns::
    dev_pt: pointwise deviance value, ndarray of shape of y_true and y_pred
    """

    assert y_true.shape == y_true.shape, "Shapes of y_true and y_pred don't match!"
    if loss_type == "poisson":
        dev_pt = 2.0 * (
            y_true * (np.log(stable(y_true)) - np.log(stable(y_pred))) + y_pred - y_true
        )
    elif loss_type == "gaussian":
        dev_pt = (y_true - y_pred) ** 2
    elif loss_type == "binominal":
        dev_pt = 2.0 * (
            -y_true * np.log(stable(y_pred))
            - (1.0 - y_true) * np.log(stable(1.0 - y_pred))
            + y_true * np.log(stable(y_true))
            + (1.0 - y_true) * np.log(stable(1.0 - y_true))
        )
    return dev_pt


def pointwise_null_deviance(y, loss_type="poisson"):
    """
    Compute pointwise null deviance for data with given loss type
    Input parameters::
    y: input data, ndarray
    loss_type: {'gaussian', 'poisson', 'binominal'}, default = 'poisson'

    Returns::
    null_dev_pt: pointwise null deviance value, ndarray of shape of y
    """
    mean_y = np.mean(y, axis=0)
    null_dev_pt = pointwise_deviance(y, mean_y, loss_type=loss_type)
    return null_dev_pt


def null_deviance(y, loss_type="poisson"):
    """
    Compute null deviance for data with given loss type, average over n_samples for each response
    Input parameters::
    y: input data, ndarray of shape (n_samples, n_responses)
    loss_type: {'gaussian', 'poisson', 'binominal'}, default = 'poisson'

    Returns::
    null_dev: average null deviance for each response, ndarray of shape of (n_responses,)
    """
    mean_y = np.mean(y, axis=0)
    null_dev = np.sum(pointwise_deviance(y, mean_y, loss_type=loss_type), axis=0)
    return null_dev


def deviance(y_pred, y_true, loss_type="poisson"):
    """
    Compute fraction deviance explained, model deviance and null deviance for data with given loss type,
    averaged over n_samples for each response
    Input parameters::
    y_pred: predicted values, ndarray of shape (n_samples, n_responses)
    y_true: true values, ndarray of shape (n_samples, n_responses)
    loss_type: {'gaussian', 'poisson', 'binominal'}, default = 'poisson'

    Returns::
    frac_dev_expl: average fraction deviance explained for each response, ndarray of shape of (n_responses,)
    d_model: average model deviance for each response, ndarray of shape of (n_responses,)
    d_null: average null deviance for each response, ndarray of shape of (n_responses,)
    """

    mean_y = np.mean(y_true, axis=0)
    d_null = np.sum(pointwise_deviance(y_true, mean_y, loss_type=loss_type), axis=0)
    d_model = np.sum(pointwise_deviance(y_true, y_pred, loss_type=loss_type), axis=0)
    frac_dev_expl = 1.0 - d_model / stable(d_null)

    if isinstance(
        frac_dev_expl, type(y_true)
    ):  # if dev is still an ndarray (skip if is a single number)
        frac_dev_expl[mean_y == 0] = (
            0  # If mean_y == 0, we get 0 for model and null deviance, i.e. 0/0 in the deviance fraction.
        )

    return frac_dev_expl, d_model, d_null


def make_prediction(X, w, w0, activation="exp"):
    """
    Make GLM prediction
    Input parameters::
    X: design matrix, ndarray of shape (n_samples, n_features)
    w: weight matrix, ndarray of shape (n_features, n_responses)
    w0: intercept matrix, ndarray of shape (1, n_responses)
    activation: {'linear', 'exp', 'sigmoid', 'relu', 'softplus'}, default = 'exp'

    Returns::
    prediction: model prediction, ndarray of shape (n_samples, n_responses)
    """
    if activation == "exp":
        prediction = np.exp(w0 + np.matmul(X, w))
    elif activation == "relu":
        prediction = np.maximum((w0 + np.matmul(X, w)), 0)
    elif activation == "softplus":
        prediction = np.log(
            stable(np.exp(w0 + np.matmul(X, w)) + 1.0)
        )  # take softplus = log(exp(features) + 1
    elif activation == "linear":
        prediction = w0 + np.matmul(X, w)
    elif activation == "sigmoid":
        prediction = 1.0 / (1.0 + np.exp(-w0 - np.matmul(X, w)))
    return prediction


def parse_group_from_feature_names(feature_names):
    """
    Parse feature_names into groups using hand-crafted rules

    Input parameters::
    feature_names: List of feature names. In this example, expanded features must contain bumpX or timeshiftX in the name

    Returns::
    group_size: list of number of features in each group
    group_name: name of each group
    group_ind: group index for each feature in feature_names, ndarray of size (len(feature_names),)es
    """
    # Find expanded features and their number of sub-features:
    group_size = list()
    group_name = list()
    group_ind = list()
    for name in feature_names:
        if "bump" not in name:
            # Non-bump expanded feature:
            group_size.append(1)
            group_name.append(name)

        elif "bump0" in name and "timeshift" not in name:
            # First bump of a bump-expanded feature:
            group_size.append(1)
            group_name.append(name[:-6])

        elif "bump" in name and "timeshift" not in name:
            # Subsequent bumps of an expanded feature (assumes that bumps are in order!):
            group_size[-1] += 1

        elif "timeshift1" in name and "bump0" in name:
            # First time shift, first bump
            group_size.append(1)
            group_name.append(name[11:-6])

        else:
            # Subsequent time shifts and bumps
            group_size[-1] += 1

    # merge all acqBlock-variables into one group
    acqBlockInd = [idx for idx, name in enumerate(group_name) if "acqBlock" in name]
    numAcqBlockVar = len(acqBlockInd)
    group_name[acqBlockInd[0]] = "acqBlock"
    group_size[acqBlockInd[0]] = numAcqBlockVar
    # if more than one acqBlockVar, delete the extra acqBlockVar groups
    if numAcqBlockVar > 1:
        del group_name[acqBlockInd[1] : acqBlockInd[-1] + 1]
        del group_size[acqBlockInd[1] : acqBlockInd[-1] + 1]

    # generate group_ind from group_size
    for i_group, this_size in enumerate(group_size):
        group_ind += [i_group] * this_size

    return group_size, group_name, np.array(group_ind)


def make_group_matrix(group_size):
    """
    Make matrix for or group scaling

    Input::
    group_size: list of size (number of features) in each group

    Returns::
    group_matrix: matrix that converts from a n_group vector to an n_expanded_group vector for scaling the groups differently,
                  ndarray of shape (n_groups, n_features), n_features = sum(group_size)
    group_ind: group indices for each feature in feature_names, list of len = n_features
    """
    group_matrix = block_diag(*[np.ones((1, n)) for n in group_size])
    return group_matrix


def make_prior_matrix(group_size):
    """
    Make prior covariance matrix that encourages smoothness in expanded features (prior)
    Based on Peron, S.P., Freeman, J., Iyer, V., Guo, C., and Svoboda, K. (2015). A Cellular Resolution Map of Barrel
    Cortex Activity during Tactile Behavior. Neuron 86, 783–799.

    Input::
    group_size: list of size (number of features) in each group

    Returns::
    prior_matrix: Block-diagonal prior matrix containing [-1, 2, 1] on the diagonal for bump-expanded features,
                  ndarray of shape (len(feature_names), len(feature_names))
    """

    def prior_component_matrix(n):
        """Create a matrix of size n * n containing the 2nd derivative of the identity matrix."""
        if n == 1:
            return 1
        else:
            return (
                np.diag(2 + np.zeros(n), k=0)
                + np.diag(-1 + np.zeros(n - 1), k=-1)
                + np.diag(-1 + np.zeros(n - 1), k=1)
            )

    prior_matrix = block_diag(*[prior_component_matrix(n) for n in group_size])

    return prior_matrix
