# Tensorflow Neural Network for solving MNIST

[Wiki](https://en.wikipedia.org/wiki/MNIST_database): MNIST is a data set of hand-written digits (28x28 pixel) taken from 
high school students and employees of the United States Census Bureau.

Useful References:
- Data and further details about the data is on [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)
- 
[TensorFlow tutorial for MNIST](https://www.tensorflow.org/get_started/mnist/beginners)

## Usage

For usage example, please see `neural_network_model.mlp_usage_example.py`. 
The main business logic is contained in `neural_network_model.mlp.py`.

Please ignore all files in folder `sandbox` as they are for prototyping. They are not used 
anywhere.

## Installation on MacOS 
Notes: 
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


