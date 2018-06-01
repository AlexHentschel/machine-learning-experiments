# Capsule Networks for MNIST in Keras

Implementation is heavily based on [Xifeng Guo's CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras).
Benchmark: [MNIST](https://en.wikipedia.org/wiki/MNIST_database)


**Additional literature recommendations:**
- Max Pechyonkin's blog post [Understanding Hinton’s Capsule Networks](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b) 
- Hinton's original paper: [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
- Huadong Liao's TensorFlow implementation [CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)

## Modifications:
- use softmax as classification error 
- use `keras.losses.binary_crossentropy` as loss 


## Installation on MacOS 
Notes: 
- Tested on Anaconda 5.0.1 x64, Python 3.6.2

### TensorFlow

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

### Optional (recommended) dependencies
- `h5py`: required if you plan on saving Keras models to disk
  ```
  $> pip install h5py
  ```
- `graphviz` and `pydot`: used by visualization utilities to plot model graphs
  ```
  $> pip install graphviz pydot
  ```
- some statistincs and machine leearning libraries: 
  ```
  $> pip install numpy scipy scikit-learn
  ```
- `pillow`: Python Imaging Library
  ```
  $> pip install pillow
  ```

### Keras
installing Keras itself:
```
$> pip install keras
```


**Notes:**
- By default, Keras will use TensorFlow as its backend. TensorFlow is also the backend I use here (for no particular reason than convenience). Switching the backend to CNTK or Theano
is documented [here](https://keras.io/backend/).
