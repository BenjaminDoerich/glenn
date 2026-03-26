import torch
import yaml
import argparse
import __init__

def main(args):
    with open(args.path_to_config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if args.mode == "Train":
        torch.set_float32_matmul_precision('high')
        from train import train_lightning
        experiment_config = config[args.config_key]
        training_config = config[args.config_key]["NN"]["training"]
        if training_config["precision"] == 32:
            print("Precision for training set to 32")
            torch.set_default_dtype(torch.float32)
            torch.set_float32_matmul_precision('high')
        if training_config["precision"] == 64:
            print("Precision for training set to 64")
            torch.set_default_dtype(torch.float64)
            torch.set_float32_matmul_precision('high')
        loading_config = config[args.config_key]["NN"]["loading_config_for_FEM"]
        FEM_solver_config = config[args.config_key]["FEM_solver_config"]
        model_config = config[args.config_key]["NN"]["model"]
        litPinn, model, train_loss = train_lightning(config[args.config_key])

    
    if args.mode == "Plot_all":
        torch.set_float32_matmul_precision('high')
        from train import plot_all
        experiment_config = config[args.config_key]
        training_config = config[args.config_key]["NN"]["training"]
        if training_config["precision"] == 32:
            print("Precision for training set to 32")
            torch.set_default_dtype(torch.float32)
            torch.set_float32_matmul_precision('high')
        if training_config["precision"] == 64:
            print("Precision for training set to 64")
            torch.set_default_dtype(torch.float64)
            torch.set_float32_matmul_precision('high')
        loading_config = config[args.config_key]["NN"]["loading_config_for_FEM"]
        FEM_solver_config = config[args.config_key]["FEM_solver_config"]
        model_config = config[args.config_key]["NN"]["model"]
        litPinn, model, train_loss = plot_all(config[args.config_key])

    if args.mode == "Refine":
        torch.set_default_dtype(torch.float64)
        torch.set_float32_matmul_precision('high')
        from train import train_lightning, refine_lightning_multiKappa
        experiment_config = config[args.config_key]
        training_config = config[args.config_key]["NN"]["training"]
        loading_config = config[args.config_key]["NN"]["loading_config_for_FEM"]
        FEM_solver_config = config[args.config_key]["FEM_solver_config"]
        model_config = config[args.config_key]["NN"]["model"]
        litPinn, model, train_loss = refine_lightning_multiKappa(config[args.config_key])

    if args.mode == "RefineFocused":
        torch.set_default_dtype(torch.float64)
        torch.set_float32_matmul_precision('high')
        from train import train_lightning, runtime_refinement_focused_kappa
        experiment_config = config[args.config_key]
        training_config = config[args.config_key]["NN"]["training"]
        loading_config = config[args.config_key]["NN"]["loading_config_for_FEM"]
        FEM_solver_config = config[args.config_key]["FEM_solver_config"]
        model_config = config[args.config_key]["NN"]["model"]
        litPinn, model, train_loss = runtime_refinement_focused_kappa(config[args.config_key])

    if args.mode == "Interpolate":
        import logging
        import FEM_solver
        import FEM_interpolate_NN
        import GL_FEM_conv_test
        experiment_config = config[args.config_key]
        training_config = config[args.config_key]["NN"]["training"]
        loading_config = config[args.config_key]["NN"]["loading_config_for_FEM"]
        FEM_solver_config = config[args.config_key]["FEM_solver_config"]
        model_config = config[args.config_key]["NN"]["model"]
        if training_config["precision"] == 32:
            print("Precision for training set to 32")
            torch.set_default_dtype(torch.float32)
            torch.set_float32_matmul_precision('high')
        if training_config["precision"] == 64:
            print("Precision for training set to 64")
            torch.set_default_dtype(torch.float64)
            torch.set_float32_matmul_precision('high')
        logger = logging.getLogger(f"lalala")
        FEM_interpolate_NN.prepare_initial_data(model_config,training_config,loading_config,FEM_solver_config, experiment_config,logger=logger)


    if args.mode == "Solve":
        import FEM_solver
        import FEM_interpolate_NN
        import GL_FEM_conv_test
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        experiment_config = config[args.config_key]


        
        training_config = config[args.config_key]["NN"]["training"]
        loading_config = config[args.config_key]["NN"]["loading_config_for_FEM"]
        FEM_solver_config = config[args.config_key]["FEM_solver_config"]
        model_config = config[args.config_key]["NN"]["model"]

        FEM_solver.run_FEM_minimzer_post(model_config,training_config,loading_config,FEM_solver_config, experiment_config)

    if args.mode == "ComputeConvergencePlot":
        import FEM_solver
        import FEM_interpolate_NN
        import GL_FEM_conv_test
        experiment_config = config[args.config_key]

        training_config = config[args.config_key]["NN"]["training"]
        loading_config = config[args.config_key]["NN"]["loading_config_for_FEM"]
        FEM_solver_config = config[args.config_key]["FEM_solver_config"]
        model_config = config[args.config_key]["NN"]["model"]
        GL_FEM_conv_test.run_GL_FEM_conv_test(model_config,training_config,loading_config,FEM_solver_config, experiment_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_config",type=str, help="Path to your config.yml file.")
    parser.add_argument("--config_key", help="The key inside of your config file which corresponds to the desired experiment that you want to run.")
    parser.add_argument("--mode", help="The mode to run: Train: train NN, Solve: run solver, Refine: refine pretrained model with 64 bit, RefineFocused: refine pretrained model with 64 bit only for loading_kappa.")

    args = parser.parse_args()

    main(args)

