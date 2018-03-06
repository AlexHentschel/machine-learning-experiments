# Tensor-Factorization based on Tensorflow for collaborative filtering

Tutorial implementation of tensor-factorization in TensorFlow. 
The implementation is inspired by the algorithm described in Albert Au Yeung
's [blogg post](http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/).
However, the implementation deviates in several aspects:
- TensorFlow is used for computing the tensor factorization through stochastic gradient
- Stochastic gradient decent is performed on mini-batches and processing a batch is implemented as matrix operations


## TensorFlow Installation on MacOS 

Please make sure that you are installing TensorFlow in an Anaconda Python distribution (or virtual envirement thereof). 

- Tested on Anaconda 5.0.1 x64, Python 3.6.2

**Install TensorFlow**

1. create new virtual env:
    ```
    $> python -m venv --symlinks --without-pip /Users/alex/Development/PythonVEs/tensorflow
    $> source /Users/alex/Development/PythonVEs/tensorflow/bin/activate
    $> curl https://bootstrap.pypa.io/get-pip.py | python
    $> deactivate
    $> source /Users/alex/Development/PythonVEs/tensorflow/bin/activate
    ```
2. Ensure pip ≥8.1 is installed:
   ```
   $> easy_install -U pip
   ```
  
3. Install TensorFlow
   ```
   $> pip3 install --upgrade tensorflow
   ```
4. Install other useful dependencies for data science
   ```
   $> pip install matplotlib pandas gzip
   ```

**Install This Package (`mnist_tensorflow`)**

Install package in developer mode (switch `-e`) 
```
$> pip install -e <path to folder containing setup.py>
```
