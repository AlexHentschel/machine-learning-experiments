# Variational Autoencoder in Keras

Implementation of Variational Autoencoder as described 
[Louis Tiao blog post](http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/) 
and [The Keras Blog](https://blog.keras.io/building-autoencoders-in-keras.html).
Benchmark: [MNIST](https://en.wikipedia.org/wiki/MNIST_database)


**Additional literature recommendations:**
- [Jaan Altosaar's Blog Post](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
- [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) by Carl Doersch
- This [whitepaper](http://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html) describes implementation in TensorFlow
- A technical overview over the different variants of Autoencoders are presented in [Bengio's paper](https://arxiv.org/pdf/1206.5538.pdf)
- Yoshua Bengio's [Deep Learning Review](http://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf) (which I generally recommend) also has a section (5.2) on autoencoders (he uses the term autoassociator).
- I haven't read [this online article](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) in details, but briefly looking over it, it also seemed ok


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


