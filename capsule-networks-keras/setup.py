from codecs import open

from setuptools import setup

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    requirements = [line.strip() for line in f]

setup(
    name='capsule_networks_keras',
    version='0.0.1',

    description='Capsule Network for solving MNIST',
    long_description=readme,
    install_requires=requirements,

    author=['Xifeng Guo', 'Alexander Hentschel'],
    author_email=['guoxifeng1990@163.com', 'alex.hentschel@axiomzen.co'],

    packages=['mnist_capnet'],
)
