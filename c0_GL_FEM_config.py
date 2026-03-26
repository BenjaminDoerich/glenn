import numpy as np
import yaml

class ConfigFEM:
    def __init__(self,FEM_solver_config):
        self.config = FEM_solver_config

        self.config["minimizer_dict"]["tol_Newton"] = float(self.config["minimizer_dict"]["tol_Newton"])
        self.config["minimizer_dict"]["tol"] = float(self.config["minimizer_dict"]["tol"])

        if self.config["spaces_config_dict"]["Fem_type_mag"] == 'Lagrange':
            self.config["spaces_config_dict"]["inc_div"] = True
            self.config["spaces_config_dict"]["corr_div"] = False

        self.problem_dict = self.config["problem_dict"]

        if self.config["spaces_config_dict"]["Fem_type_mag"] == 'Lagrange':
            self.config["minimizer_dict"]["Newton_extended"] = False

        self.minimizer_dict = self.config["minimizer_dict"]

        MS_value = self.config["spaces_config_dict"]["Ms_value"]
        M_ref = self.config["spaces_config_dict"]["M_ref"]

        self.config["spaces_config_dict"]["has"] = 4 / MS_value
        self.config["spaces_config_dict"]["h_ref"] = 4 / M_ref

        self.spaces_config_dict = self.config["spaces_config_dict"]

        self.use_rand = True
        self.use_rand = False


    def dump_config(self, results=""):
        self.config["results"]=results
        with open(self.config["output_path"] + "/" + self.config["results_filename"] + ".yml", "w") as outfile:
            yaml.dump(self.config, outfile,default_flow_style=False)







