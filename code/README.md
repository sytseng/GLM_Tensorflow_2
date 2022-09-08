# Code for GLM class
The Python script `glm_class.py` contains GLM class definitions and utility functions for fitting GLM in Tensorflow 2. See the notebook `Tutorial_for_using_GLM_class.ipynb` in the [tutorial](https://github.com/sytseng/GLM_Tensorflow_2/tree/main/tutorial) folder for the guide for using this code.

## Quick start
Load the code as a module 
```
import glm_class as glm
```

Initialize a GLM (without cross validation)
```
model = glm.GLM(activation = 'linear', loss_type = 'gaussian', 
                regularization = 'elastic_net', lambda_series = 10.0 ** np.linspace(3, -6, 10), l1_ratio = 0., smooth_strength = 0., 
                optimizer = 'adam', learning_rate = 1e-2, momentum = 0.5, 
                min_iter_per_lambda = 100, max_iter_per_lambda = 10**4, num_iter_check = 100, convergence_tol = 1e-6)
```

Fit the model on training data
```
model.fit(X, Y, [initial_w0, initial_w, feature_group_size, verbose])
```

Model selection over regularization strengths (lambda_sereis) with validation data
```
model.select_model(X_val, Y_val, [min_lambda, make_fig])
```

Evaluate model performance (on training, validation, or test data)
```
frac_dev_expl, dev_model, dev_null, dev_expl = model.evaluate(X, Y, [make_fig])
```

Obtain selected weights and intercepts
```
model.selected_w
model.selected_w0
```

Obtain selected regularization strengths and indices
```
model.selected_lambda
model.selected_lambda_ind
```

Obtain loss during fitting and the associated lambda values
```
model.loss_trace
model.lambda_trace
```
