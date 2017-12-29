# Variational Autoencoder in Keras

Implementation of **V**ariational **A**uto**e**ncoder (**VAE**) as described in
[Louis Tiao's blog post](http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/) 
and [The Keras Blog](https://blog.keras.io/building-autoencoders-in-keras.html).
Benchmark: [MNIST](https://en.wikipedia.org/wiki/MNIST_database)


**Additional literature recommendations:**
- Carl Doersch wrote an **excellent paper introducing the mathematical background**: [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) 
- This [whitepaper](http://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html) describes VAE implementation in TensorFlow
- [Jaan Altosaar's Blog Post](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) describes the two different views on VAEs: from _deep learning_ perspective and _probabilistic generative model_ perspective
- A technical overview over the different variants of Autoencoders are presented in [Bengio's paper](https://arxiv.org/pdf/1206.5538.pdf)
- Yoshua Bengio's [Deep Learning Review](http://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf) (which I generally recommend) also has a section (5.2) on autoencoders (he uses the term autoassociator).

**Recommendations on Variational Inference:**
- David Blei's [course notes](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf) and his paper [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670) 


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
- By default, Keras will use TensorFlow as its backend. Switching the backend to CNTK or Theano
is documented [here](https://keras.io/backend/).