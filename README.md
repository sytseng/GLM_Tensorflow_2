# GLM_Tensorflow_2
Code and tutorials for fitting generalized linear models (GLM) in Tensorflow 2.

Written by [Shih-Yi Tseng](https://github.com/sytseng) from the [Harvey Lab](https://harveylab.hms.harvard.edu/) at Harvard Medical School, with special acknowledgements to [Matthias Minderer](https://github.com/mjlm) and [Selmaan Chettih](https://github.com/Selmaan).


## Structure of this repository:
- The Python script `GLM_class.py` in the [code](https://github.com/sytseng/GLM_Tensorflow_2/tree/main/code) folder contains GLM class definitions and utility functions for fitting GLM in Tensorflow 2.
- The two notebooks `Tutorial_for_using_GLM_class.ipynb` and `Tutorial_for_fitting_neural_calcium_imaging_data_with_GLM.ipynb` in the [tutorial](https://github.com/sytseng/GLM_Tensorflow_2/tree/main/tutorial) folder contain tutorials for how to use the GLM_class code and how to fit neural calcim imaging data with GLMs, respectively.
- The Pickle file `example_data_glm.pkl` in the [data](https://github.com/sytseng/GLM_Tensorflow_2/tree/main/data) folder contains example calcium imaging data used in the second tutorial.

## Software requirements
The code was developed with the following Python packages:
- numpy version 1.21.6
- scipy version 1.7.3
- sklearn version 1.0.2
- tensorflow version 2.8.2
- keras version 2.8.0
- matplotlib version 3.2.2

It works most efficiently on a GPU. Note that eager execution for Tensorflow must be enabled to run the code.

## References:
- Tseng, S.-Y., Chettih, S.N., Arlt, C., Barroso-Luque, R., and Harvey, C.D. (2022). Shared and specialized coding across posterior cortical areas for dynamic navigation decisions. Neuron 110, 2484–2502.e16. [[link]](https://www.sciencedirect.com/science/article/pii/S0896627322004536) 
- Minderer, M., Brown, K.D., and Harvey, C.D. (2019). The spatial structure of neural encoding in mouse posterior cortex during navigation. Neuron 102, 232–248.e11. [[link]](https://www.sciencedirect.com/science/article/pii/S089662731930056X)
