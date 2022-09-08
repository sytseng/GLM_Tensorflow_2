# Code for GLM class
The Python script `glm_class.py` contains GLM class definitions and utility functions for fitting GLM in Tensorflow 2. See the notebook `Tutorial_for_using_GLM_class.ipynb` in the [tutorial](https://github.com/sytseng/GLM_Tensorflow_2/tree/main/tutorial) folder for a comprehensive guide for using this code.

## Quick start
Load the code as a module 
```
import glm_class as glm
```

### Initialize a GLM (without cross validation)
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

3. Loss during fitting and the associated lambda values
```
model.loss_trace
model.lambda_trace
```


### Initialize a GLM_CV (GLM with cross validation)
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

Same `evaluate`, `predict`, and model attributes as GLM class.
