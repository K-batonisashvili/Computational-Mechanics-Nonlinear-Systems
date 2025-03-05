# Assignment # 2.2 Direct and Geometric Stiffness Method
This assignment focuses on implementing the Direct and Geometric Stiffness Method for structural analysis. We have created a class-based structure to model nodes, elements, and frames, and to perform stiffness matrix calculations, including geometric stiffness analysis to determine the elastic critical load factor.

This repo contains this README, Direct_Stiffness.py, test_geometricstiffness.py, tutorials.ipynb, assignment_2_2.ipynb, pyproject.toml, and the GenAIUSE.txt.

- `README.md`: This file, containing instructions.
- `Geometric_stiffness.py`: Contains the main code for the Direct and Geometric Stiffness Method.
- `test_geometricstiffness.py`: Contains the test functions to be used with Pytest.
- `tutorials.ipynb`: The Jupyter notebook containing guides.
- `assignment_2_2.ipynb`: The Jupyter notebook containing the assignment.
- `genAIuse.txt`: Contains the statement describing how AI was used for this assignment.
- `pyproject.toml`: Contains all the requirements for getting the library setup.

## Getting Started
When cloning this repo, you will have a general `src` (source) folder with all the required files within it. `README.md` and `pyproject.toml` will be in the base directory next to the `src` folder and that is normal. The rest of the setup for this should be fairly straightforward.

We are going to create a new conda environment for this code. The reason being it is easier to keep track of all dependencies for how this runs within that conda environment. That way if you need to adjust something on your machine in the future, these installed packages will not mess with anything.

If you do not have conda installed on your computer, I will have a generic conda installation tutorial in the future. For now, please follow the commands below to create your conda environment.

## Conda Environment Setup

Begin by setting up a conda or mamba environment:
```bash
conda create --name geometric-stiffness-env python=3.12
```
Once the environment has been created, activate it:

```bash
conda activate geometric-stiffness-env
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

For the more advanced, you may change the code manually within Direct_Stiffness.py and test_directstiffness.py based on whatever needs you may have.