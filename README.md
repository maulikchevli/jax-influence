# Jax Influence

Scalable implementation of Influence Functions in JaX.

Implementation of the algorithms in
[Scaling Up Influence Functions (AAAI 2022)](https://arxiv.org/abs/2112.03052)
for efficient calculation of Influence Functions.

## Installation
To install `jax_influence` as a pip module, just run `pip install .` after installing the requirements mentioned in [environment.yml](./environment.yml).

With conda, the following [installation](./install.sh) script can be used to install everything (for CPU) with `source install.sh`.

### Conda installation script
For CPU only installation:
```bash
conda create -n jax-influence python=3.10.8
conda activate jax-influence
conda install pip
pip install jax jaxlib ipykernel
conda env update -n jax-influence --file environment.yml
pip install .
```
Or just run the [installation](./install.sh) script as `source install.sh` 

<br>


For GPU version, please install the appropriate Cuda toolkit and jax version from [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) and [here](https://jax.readthedocs.io/en/latest/installation.html) resp. The remaining steps remain similar to the above script. Here is an example that is compatible with Cuda11:
```sh
conda create -n jax-influence python=3.10.8
conda activate jax-influence
conda install cuda -c nvidia/label/cuda-11.8.0
conda install pip
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install ipykernel
conda env update -n jax-influence --file environment.yml
pip install .
```

### A reproducible conda environment
[jax-influence.environment.yml](./jax-influence.environment.yml): Compatible only with CUDA 11
```sh
conda create -n jax-influence python=3.10.8
conda activate jax-influence
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda env update -n jax-influence --file jax-influence.environment.yml
pip install .
```

## Documentation

An end-to-end example of using the library can be found in
`examples/colab/mnist_tutorial.ipynb`. We plan to add more examples in the
future.

## Disclaimer

This is not an official Google product.

Jax Influence is a research project, and under active development by a
small team; we'd love your suggestions and feedback - drop us a
line in the [issues](https://github.com/google-research/jax-influence).

