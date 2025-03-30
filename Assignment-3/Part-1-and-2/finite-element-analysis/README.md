# Assignment # 3.1 Tutorial for Finite Element Analysis
This assignment focuses on Finite Element Analysis on different frames and elements. There are pre-existing discretization functions that were created to assist with this analysis. Assignment 3.1 focuses on providing an in depth tutorial on how to use those functions and see element transformations. 

`Please use the Kote_Tutorial folder for the Jupyter Notebook`

This repo contains this README, src folder, tutorials folder, Kote_Tutorial folder, test folder, pyproject.toml, and the GenAIUSE.txt.

- `README.md`: This file, containing instructions.
- `src folder`: Contains the main code for FEA tools such as discretization.
- `tests folder`: Contains the test functions to be used with Pytest.
- `tutorials folder`: Jupyter notebooks containing original guides without much description.
- `Kote_Tutorial folder`: Jupyter notebook containing more in depth description on how to use the discretization helper functions.
- `genAIuse.txt`: Contains the statement describing how AI was used for this assignment.
- `pyproject.toml`: Contains all the requirements for getting the library setup.

## Getting Started
When cloning this repo, you will have a general `src` (source) folder with all the required files within it. `README.md` and `pyproject.toml` will be in the base directory next to the `src` folder and that is normal. The rest of the setup for this should be fairly straightforward.

We are going to create a new conda environment for this code. The reason being it is easier to keep track of all dependencies for how this runs within that conda environment. That way if you need to adjust something on your machine in the future, these installed packages will not mess with anything.

If you do not have conda installed on your computer, I will have a generic conda installation tutorial in the future. For now, please follow the commands below to create your conda environment.

## Conda Environment Setup

Begin by setting up a conda or mamba environment:
```bash
conda create --name finite-element-analysis-env python=3.12.9
```
Once the environment has been created, activate it:

```bash
conda activate finite-element-analysis-env
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

### Running the Jupyter Notebook

If the Jupyter notebook is running into issues and the errors say "unable to find library", that might mean one of 2 things, please try both fixes.

1) Make sure your conda environment is activated in the terminal by doing conda activate finite-element-analysis-env. Then run the command `conda list` which should include the finiteelement library.
2) When running in VScode directly, please make sure that your interpreter for the Jupyter Notebook is running the conda environment instance. In VScode, press CTRL SHIFT P and type in "interpreter" and go through the menu to select your conda env.