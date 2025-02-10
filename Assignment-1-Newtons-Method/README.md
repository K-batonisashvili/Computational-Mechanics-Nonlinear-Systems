# Assignment # 1 Newtonian Method
This is the continuation of the first assignment where we configured bisection method, except now we are utilizing the newtonian variant of that.
This repo contains this README, newton_method.py, test_newton.py, tutorial.ipynb, pyproject.toml, and the GenAIUSE.txt.

Readme.md is this file, containing instructions.

newton_method.py contains the central math portion of the newtonian algorithm.

test_newton.py contains the test functions to be used with Pytest.

tutorial.ipybn is the jupyter notebook containing guides.

genAIuse.txt contains the statement describing how AI was used for this assignment.

pyproject.toml contains all the requirements for getting the library setup.


## Getting Started
When cloning this repo, you will have a general src (source) folder with all the required files within it. readme and pyproject.toml will be in the base directory next to the src folder and that is normal. The rest of setup for this should be fairly straightforward.

We are going to create a new conda environment for this code. The reason being it is easier to keep track of all dependencies for how this runs within that conda environment. That way if you need to adjust something on your machine in the future, these installed packages will not mess with anything.

If you do not have conda installed on your computer, I will have a generic conda installation tutorial in the future. For now, please do the following commands do create your conda library.

## Conda Library Setup

Begin by setting up a conda or mamba environment:
```bash
conda create --name newton_method_env python=3.12
```
Once the environment has been created, activate it:

```bash
conda activate newton_method_env
```
Double check that python is version 3.12 in the environment:
```bash
python --version
```
Ensure that pip is using the most up to date version of setuptools:
```bash
pip install --upgrade pip setuptools wheel
```
With this pip install command, we are downloading all required packages and dependencies into your conda environment:
```bash
pip install -e .
```
Thats it!

### Code Usage
There are several different ways that you are encouraged to interact with the code.

1. Jupyter notebook

The included Jupyter notebook is a tutorial notebook which guides you step by step on how the test functions are created, how some cases pass and fail, and how you are able to create your very own tests.

2. Manually Editing Code

For the more advanced, you may change the code manually within test_newton.py or newton_method.py. There are 2 values which are user-editible. **TOL** and **ITER** which determine to how many decimal places you would like to have a tolerance to, and how many total iterations you would like to run. Both of these adjustments contribute to balancing completion speed and accuracy. 