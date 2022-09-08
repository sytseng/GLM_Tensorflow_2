# Code for GLM class
The Python script `glm_class.py` contains GLM class definitions and utility functions for fitting GLM in Tensorflow 2. See the notebook `Tutorial_for_using_GLM_class.ipynb` in the [tutorial](https://github.com/sytseng/GLM_Tensorflow_2/tree/main/tutorial) folder for a comprehensive guide for using this code.

## Quick start
Load the code as a module 
```
import glm_class as glm
```

### GLM class (GLM without cross validation)
#### Initialize a GLM
```
model = glm.GLM(activation = 'linear', loss_type = 'gaussian', 
                regularization = 'elastic_net', lambda_series = 10.0 ** np.linspace(3, -6, 10), 
                l1_ratio = 0., smooth_strength = 0., 
                optimizer = 'adam', learning_rate = 1e-2, momentum = 0.5, 
                min_iter_per_lambda = 100, max_iter_per_lambda = 10**4, 
                num_iter_check = 100, convergence_tol = 1e-6)
```

#### Core methods
1. Fit the model on training data
    ```
    model.fit(X, Y, [initial_w0, initial_w, feature_group_size, verbose])
    ```

2. Model selection over regularization strengths (lambda_sereis) with validation data
    ```
    model.select_model(X_val, Y_val, [min_lambda, make_fig])
    ```

3. Evaluate model performance (on training, validation, or test data)
    ```
    frac_dev_expl, dev_model, dev_null, dev_expl = model.evaluate(X, Y, [make_fig])
    ```

4. Make prediction (on training, validation, or test data)
    ```
    model.predict(X)
    ```

#### Important model attributes
1. Selected weights and intercepts
    ```
    model.selected_w
    model.selected_w0
    ```

2. Selected regularization strengths and indices
    ```
    model.selected_lambda
    model.selected_lambda_ind
    ```

3. Fraction deviance explained on validation data for selected models
    ```
    model.selected_frac_dev_expl_val
    ```

4. Loss during fitting and the associated lambda values
    ```
    model.loss_trace
    model.lambda_trace
    ```

### GLM_CV class (GLM with cross validation)
#### Initialize a GLM_CV
```
model_cv = glm.GLM_CV(n_folds = 5, auto_split = True, split_by_group = True, split_random_state = 42,
                      activation = 'linear', loss_type = 'gaussian', 
                      regularization = 'elastic_net', lambda_series = 10.0 ** np.linspace(3, -6, 10), 
                      l1_ratio = 0., smooth_strength = 0., 
                      optimizer = 'adam', learning_rate = 1e-2, momentum = 0.5, 
                      min_iter_per_lambda = 100, max_iter_per_lambda = 10**4, 
                      num_iter_check = 100, convergence_tol = 1e-6)

```

#### Core methods
1. Fit the model on training data
    ```
    model_cv.fit(X, Y, [train_idx, val_idx, group_idx, initial_w0, initial_w, 
                        feature_group_size, verbose])
    ```

2. Model selection over regularization strengths based on CV held-out data
    ```
    model_cv.select_model([se_fraction, min_lambda, make_fig])
    ```

3. Evaluate model performance (on training, validation, or test data); same as GLM class
    ```
    frac_dev_expl, dev_model, dev_null, dev_expl = model_cv.evaluate(X, Y, [make_fig])
    ```

4. Make prediction (on training, validation, or test data); same as GLM class
    ```
    model_cv.predict(X)
    ```

5. Make prediction using CV fold-specific weights of selected models on CV held-out data

   Used in quantification of feature contribution with "model breakdwon" procedure
   
   X must be some variants of X_train (same samples with some features altered, e.g. zeroed or shuffled) 
    ```
    model_cv.make_prediction_cv(X)
    ```


#### Important model attributes
1. Selected weights and intercepts; same as GLM class
    ```
    model_cv.selected_w
    model_cv.selected_w0
    ```

2. Selected regularization strengths and indices; same as GLM class
    ```
    model_cv.selected_lambda
    model_cv.selected_lambda_ind
    ```

3. Fraction deviance explained on CV held-out data for selected models (with CV fold-specific weights)
    ```
    model_cv.selected_frac_dev_expl_cv
    ```

### Utility funcitons
1. Make prediction given design matrix, weights and intercepts, and activation function
    ```
    def make_prediction(X, w, w0, activation = 'exp'):
        '''
        Make GLM prediction
        Input parameters::
        X: design matrix, ndarray of shape (n_samples, n_features)
        w: weight matrix, ndarray of shape (n_features, n_responses)
        w0: intercept matrix, ndarray of shape (1, n_responses)
        activation: {'linear', 'exp', 'sigmoid', 'relu', 'softplus'}, default = 'exp'

        Returns::
        prediction: model prediction, ndarray of shape (n_samples, n_responses)
        '''
    ```

2. Compute deviance for each response variable with model prediction
    ```
    def deviance(mu, y, loss_type = 'poisson'):
        '''
        Compute fraction deviance explained, model deviance and null deviance for data with given loss type, 
        averaged over n_samples for each response 
        Input parameters::
        mu: predicted values, ndarray of shape (n_samples, n_responses)
        y: true values, ndarray of shape (n_samples, n_responses)
        loss_type: {'gaussian', 'poisson', 'binominal'}, default = 'poisson'

        Returns::
        frac_dev_expl: average fraction deviance explained for each response, ndarray of shape of (n_responses,)
        d_model: average model deviance for each response, ndarray of shape of (n_responses,)
        d_null: average null deviance for each response, ndarray of shape of (n_responses,)
        '''
    ```
    
3. Compute null deviance for each response variable
    ```
    def null_deviance(y, loss_type = 'poisson'):
        '''
        Compute null deviance for data with given loss type, average over n_samples for each response 
        Input parameters::
        y: input data, ndarray of shape (n_samples, n_responses)
        loss_type: {'gaussian', 'poisson', 'binominal'}, default = 'poisson'

        Returns::
        dev: average null deviance for each response, ndarray of shape of (n_responses,)
        '''
    ```
    
4. Compute deviance for each response variable with model prediction for each datapoint
    ```
    def pointwise_deviance(y_true, y_pred, loss_type = 'poisson'):
      '''
      Compute pointwise deviance for data with given loss type 
      Input parameters::
      y_true: true values, ndarray
      y_pred: predicted values, ndarray
      loss_type: {'gaussian', 'poisson', 'binominal'}, default = 'poisson'

      Returns::
      dev: pointwise deviance value, ndarray of shape of y_true and y_pred
      '''
    ```
    
5. Compute null deviance for each response variable for each datapoint
    ```
    def pointwise_null_deviance(y, loss_type = 'poisson'):
      '''
      Compute pointwise null deviance for data with given loss type 
      Input parameters::
      y: input data, ndarray
      loss_type: {'gaussian', 'poisson', 'binominal'}, default = 'poisson'

      Returns::
      dev: pointwise deviance value, ndarray of shape of y
      '''
    ```
