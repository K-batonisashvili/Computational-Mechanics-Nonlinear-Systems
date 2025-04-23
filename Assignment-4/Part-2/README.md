# Assignment # 4.2 FEniCSx Limits and Examples
This assignment focuses on the different components of FEniCSx, its adjustability, and its failures. FEniCSx is an open-source computing platform that may be utilized for many different scientific simulations, such as PDE's, FEA, MSA, etc. Please read through this README to understand how to fully integrate FEniCSx on your system from scratch, then head over to the Tutorial Jupyter notebook which goes shows how to set up an example problem, perform computations, and visualize the results. We will be creating a simple bridge which will have a downward force applied to it. For a full idea of the example problem, please read the tutorial. 

This is a simple repo which contains this README, FEniCSx_Tutorial.ipynb, pyproject.toml, and the GenAIUSE.txt.

- `README.md`: This file, containing instructions.
- `FEniCSx_Analytical.ipynb`: This tutorial compares the computed solutions of the program to analytical solutions.
- `FEniCSx_Failure.ipynb`: Examples and parameters to change to see when Fenicsx fails.
- `FEniCSx_p_h_Mesh_Example.ipynb`: Compares the difference between fine mesh refinement to higher order p refinement.
- `....gif`: Several different gifs which show respective deformation for each tutorial.
- `genAIuse.txt`: Contains the statement describing how AI was used for this assignment.
- `pyproject.toml`: Contains all the requirements for getting the library setup.

## Getting Started & SCC
We are going to create a new conda environment for this code. The reason being it is easier to keep track of all dependencies for how this runs within that conda environment. That way if you need to adjust something on your machine in the future, these installed packages will not mess with anything.

Conda should already be installed during your SCC session. There is a chance you have `mamba` as opposed to `conda`, in which case please replace any of the future commands that start with `conda` into `mamba`.

## Conda Environment Setup

Begin by loading the miniconda (or mamba) environment on to your SCC:
```bash
module load miniconda
```

Next, we create a new environment for FEniCSx. (If FEniCSx is already installed and you have an existing conda environment set-up, please ignore the rest of this and activate your FEniCSx conda env and skip to Visual Studio interpreter set up).
```bash
conda create -n fenicsx-env
```

Once the environment has been created, activate it:
```bash
conda activate fenicsx-env
```

Install FEniCSx, mpich, pyvista, and DolfinX through conda:
```bash
install -c conda-forge fenics-dolfinx mpich pyvista
```

Install the 3 supplemental libraries that we need to work with:
```bash
pip install imageio
pip install gmsh
pip install PyYAML
```

Thats it! 
Note: It might take a few minutes to install all the libraries.
Note2: We are not using pyproject.toml as we are operating under the assumption that we are running this code on the BU SCC where the core libraries listed above are the only ones we need. 

### Jupyter Notebook Interpreter Setup

If the Jupyter notebook is running into issues and the errors say "unable to find library", that might mean one of 2 things, please try both fixes.

1) Make sure your conda environment is activated in the terminal by doing conda activate fenicsx-env. Then run the command `conda list` which should include the FenicsX-Env library.
2) When running in VScode directly, please make sure that your interpreter for the Jupyter Notebook is running the conda environment instance. In VScode, press CTRL SHIFT P and type in "interpreter" and go through the menu to select your conda env.