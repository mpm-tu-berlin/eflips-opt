# eflips-opt
Experimental optimization tools based on MPM eflips products. 

# Installation
1. Clone the repository
    ```bash 
    git clone git@github.com:mpm-tu-berlin/eflips-opt.git
    ```

2. Install the packages listed in `poetry.lock` and `pyproject.toml` into your Python environment. Notes:
    - The suggested Python version os 3.11.*, it may work with other versions, but this is not tested.
    - Using the [poetry](https://python-poetry.org/) package manager is recommended. It can be installed according to the
      instructions listed [here](https://python-poetry.org/docs/#installing-with-the-official-installer).

3. In order to use the solver, it is recommended to install the [Gurobi](https://www.gurobi.com/) solver. The solver can be
   installed by following the instructions listed [here](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer). 
   The solver can be used with a license for academic purposes. If you do not have a license, you can request one [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

