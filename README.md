# GLENN

## Conda environment for the FE-solver and interpolation:
~~~bash
conda create --name gle-nn 
conda install -c conda-forge fenics-dolfinx mpich pyvista scipy numpy matplotlib pandas pytorch python-gmsh adios4dolfinx
pip install pykan scikit-learn pyyaml tqdm
~~~



## Python virtual env for training the NNs:
~~~bash
python3.11 -m venv "GLENN"
source GLENN/bin/activate
python -m pip install --upgrade pip
pip3 install torch==2.9 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install tensorboard lightning scipy pyyaml matplotlib
~~~



## Reproducing the experiments:
### Configuration files:
For each of the experiments we provide the necessary configurations. In the following, you find a table mapping the neural networks used in the experiments to the corresponding config file.


Any configuration beyond what is provided in our configuration files is to be treated as unfinished and untested. 

|  Model | GLENN-R1  | GLENN-R2  | GLENN-F1  | GLENN-F2  | GLENN-L*  | GLENN-U* |
|---|---|---|---|---|---|---|
| File  | config_GLENN-R1.yml  | config_GLENN-R2.yml  | config_GLENN-F1.yml  | config_GLENN-F2.yml  |  config_GLENN-L*.yml | config_GLENN-U*.yml |

The star is to be replaced with * = 1,...,5.

### Running the experiments:
- The code for training a NN is executed by using
  ~~~bash
  python main.py --path_to_config <config_file_name> --config_key <top_key_in_config_file> --mode Train
  ~~~
  with the appropriate config file and top level key, e.g.:
  ~~~bash
  python main.py --path_to_config config_GLENN-R1.yml --config_key GLENN-R1 --mode Train
  ~~~


- Analogously, the interpolation can be done using:
  ~~~bash
  python main.py --path_to_config <config_file_name> --config_key <top_key_in_config_file> --mode Interpolate
  ~~~
  Here, under the key "loading_config_for_FEM", the kappa for which the neural network is interpolated has to be adapted prior to interpolation. Additionally, under the key "problem_dict", the kappa value has to match this value.



- The solver is run via:
  ~~~bash
  mpirun -np <number_of_processors> python main.py --path_to_config <config_file_name> --config_key <top_key_in_config_file> --mode Solve
  ~~~
  For the heuristic initial values, set "use_NN_initial_guess" to False and "u_type" to 11,12,13,14, or 15, corresponding to $\varphi_1,...,\varphi_5$. For NN initial values, one has to interpolate a NN first as described in the previous point. Under the key "problem_dict", the value for kappa that the solver uses has to be adapted to the desired parameter.



  For example, to solve with 16 processes using a previously interpolated GLENN-R1 checkpoint, one should use:
  ~~~bash
  mpirun -np 16 python main.py --path_to_config config_GLENN-R1.yml --config_key GLENN-R1 --mode Solve
  ~~~