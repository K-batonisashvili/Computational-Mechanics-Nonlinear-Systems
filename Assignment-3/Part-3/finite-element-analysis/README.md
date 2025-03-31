# Assignment # 3.3 High-level Overview and Tutorial of FEA Files and Functions
This assignment focuses on providing an in-depth, high-level overview of the Finite Element Analysis files, functions, concepts, and overall code. The beginning of this tutorial will go through the basics on downloading the files and creating a virtual conda environment to ensure everything runs smoothly. Once this first step is complete, this README will dive into how to use the code properly. 

Assignment 3 part 3 contains many different files and folders. Most noteable of them are described below:

- `README.md`: This file, containing instructions.
- `src folder`: Contains the main code for FEA tools such as discretization.
- `tests folder`: Contains the test functions to be used with Pytest.
- `tutorials folder`: Jupyter notebooks containing strictly code guides, python tutorial files on how to import and utilize the code, and graphic outputs from the tutorials.
- `genAIuse.txt`: Contains the statement describing how AI was used for this assignment.
- `pyproject.toml`: Contains all the requirements for getting the library setup.

## Getting Started
When cloning this repo, you will have a general `src` (source) folder with all the required files within it. `README.md` and `pyproject.toml` will be in the base directory next to the `src` folder. `Tutorials` folder will be our primary focus which will have sections that will peer into the other parts. The rest of the setup for this should be fairly straightforward.

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
If you have a folder from which you are working from, please navigate there now. Otherwise, create a new folder to clone this repo and then navigate into it:
```bash
mkdir Finite-Element-Analysis
cd Finite-Element-Analysis
```
We need to git clone the original repo and then navigate to the FEA Part 3 sub-directory to install all the pre-requisites. In your terminal, please write the following:
```bash
git clone https://github.com/K-batonisashvili/Computational-Mechanics-Nonlinear-Systems.git
```
Once you clone the directory, lets navigate to the appropriate sub-directory. Your path should look similar to this:
```bash
cd ./Assignment-3/Part-3/finite-element-analysis
```
Finally, with this pip install command, we will download all required packages and dependencies into your conda environment:
```bash
pip install -e .
```
Thats it!

### TROUBLESHOOTING: If you encounter "finiteelementanalysis module not available"

If any of the code is running into issues and the errors which say "unable to find library", that might mean one of 2 things, please try both fixes.

1) Make sure your conda environment is activated in the terminal by doing conda activate finite-element-analysis-env. Then run the command `conda list` which should include the finiteelementanalysis library.
2) When running in VScode directly, please make sure that your interpreter for the Jupyter Notebook is running the conda environment instance. In VScode, press CTRL SHIFT P and type in "interpreter" and go through the menu to select your conda env.


## Tutorial Deep Dive

In this part of the README, we will heavily dive into the tutorial section of our Finite Element Analysis code, which will branch out to the remaining sections. If there is mention of images or plotted graphics and you do not see those in your directory, that would be because those graphics were not included in the github repo due to size limitations. Stepping through the tutorial files should generate those graphics.

### full_code_example_1.py

Let us begin by examining full_code_example_1.py. This is a python file acting as a tutorial on how to import all the necessary modules and functions, how to initialize all the required variables for analysis, perform the mathematical computations, and finally generate graphics to assist with visualization. We will go through an overall breakdown of this file, and then discuss the supplemental code which can be found in the `src` folder.

The first part of this code imports all computational functions from our finiteelementanalysis module, matplotlib for plotting, and system modules for file saving and warnings. If you run into any issues which say "finiteelementanalysis module not found" or "not available", a few sections above are some hints that can help with troubleshooting. 

In this tutorial, we will be analyzing a rectangular shape using triangular elements. This rectangle is discretized into several sections to perform an analysis with these triangular elements in a mesh, with resulting data (numeric and graphic) showing the effects on displacement when a load is applied. From lines 10 through 70, we initialize all our initial variables such as element type, the degrees of freedom, rectangle dimensions, and more. The exact details of the entire shape is seen in the comments of the file. 

Once the shape and its respective properties are defined, lines 70 through 99 perform the computation in finding the solution. This may be checked directly against an analytical solution to ensure that our code works properly. 

While there isn't a numerical implementation of checking to see if the answers match, there exists a png called `uniaxial_extension_error.png` which plots the solutions of the analytical answers to the computed ones, and that should be located in the same directory. Lines 99 through 117 are what provide the visual confirmations and plots on how the rectangular frame experiences load. 