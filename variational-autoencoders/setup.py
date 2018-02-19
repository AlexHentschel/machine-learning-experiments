from codecs import open

from setuptools import setup

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    requirements = [line.strip() for line in f]

setup(
    name='variational-autoencoders',
    version='0.0.1',

    description='Tensorflow Neural Network for solving MNIST',
    long_description=readme,
    install_requires=requirements,

    author=['Alexander Hentschel'],
    author_email=['alex.hentschel@axiomzen.co'],

    packages=['variational_autoencoders'],
)
