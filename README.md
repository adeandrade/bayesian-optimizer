# Bayesian Optimization Package

A Bayesian Optimizer model that uses an ARD Matérn 3/2 Kernel and Expected Improvement as the acquisition function.

Similarly to Spearmint, initial candidates are drawn from a Sobol sequence, then a subset of points with the highest acquisition score get optimized against Expected Improvement using L-BFGS-B.

With each new trial, the gaussian process is optimized against the kernel parameters: a multiplying constant, and the characteristic length scale of each dimension (defining diagonal covariances). The prior mean of the gaussian process is assumed to be constant and zero.
 
The package defines a AWS S3 storage engine for the model. Please configure `storage_location` inside `store.py` to point to the desired location of your data. The `Interface` class allows you to interact with multiple models in an offline fashion, but it also serves as an example to create an online interface to this model.

The code is only compatible with Python >= 3.7.0. Package requirements are specified in `requirements.txt`
