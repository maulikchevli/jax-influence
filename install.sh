conda create -n jax-influence python=3.10
conda activate jax-influence
conda install pip
pip install jax jaxlib ipykernel
conda env update -n jax-influence --file environment.yml
pip install .