#--------------------Libraries--------------------#
#hydra
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
#wandb
import wandb
#pytorch
import torch
#dealing with files and paths etc.
import os
import itertools
import json
from copy import deepcopy
#data handling
import pandas as pd
import numpy as np
#--------------------Functions--------------------#
#functions for converting search space from yaml to list of concrete model configs for sweeping
def expand_search_space_dim(space_dict):
    #conerting DictConfig to regular dict for easier handling and logging in wandb (fails downstream without it))
    space_dict = OmegaConf.to_container(space_dict, resolve=True)
    #prepare keys and values (variables) for expansion
    keys = list(space_dict.keys())#this gets names of all parameters (for example learning rate, momentum, regularization strength, optimizer, etc.) as a list
    values = []
    #Putting all values as lists (if not already lists) for easier expansion using itertools.product (which expects iterables for expansion, i.e. lists, not single values which might be the case)
    for value in space_dict.values():#iterating over values in different keys
        if isinstance(value, (list, tuple)):#if list or tuple, keep as list
            values.append(list(value))
        else:#if not make one-element list
            values.append([value])
    #prepare varaible to hold each configuration (combination of parameters) as a dict for easier handling and logging
    results = []
    #expanding using cartesian product of all values (all combinations of parameters) and creating dict for each combination with keys as parameter names and values as specific value for that parameter in that combination
    for combo in itertools.product(*values):
        #adding single configuration (combination of parameters) as a dict to results list
        results.append(dict(zip(keys, combo)))
    #returning output
    return results
def expand_search_space(search_cfg):
    #getting all pre-processing variants
    preprocess_runs = expand_search_space_dim(search_cfg.preprocess)
    #preparing list to hold all combinations of model and pre-processing configurations for sweeping
    runs = []
    #iterating through all models
    for model_name, model_space in search_cfg.models.items():
        #getting all variants for current model
        model_runs = expand_search_space_dim(model_space)
        #combining each model configuration with each pre-processing configuration and adding to runs list
        for model_params in model_runs:#iterating through all model params combinations for current model
            for preprocess_params in preprocess_runs:#iterating through all pre-processing params combinations
                #joining current model configuration with current pre-processing configuration and adding to runs list
                runs.append(
                    {
                        "model": {
                            "name": model_name,#name of the model implemented in model.py
                            "params": model_params,#hypermaeters for model training
                        },
                    }
                )
    #returning output
    return runs
#function for starting wandb run 
def start_wandb_run(cfg: DictConfig, run_cfg: dict, job_type: str):
    #starting wandb run for current configuration
    run = wandb.init(
        #setting up wandb info for run identification and organization in the dashboard
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group=cfg.logger.get("group", None),
        mode=cfg.logger.get("mode", "online"),
        job_type=job_type,
        #config part of wandb info
        config={
            "seed": cfg.seed,#no need for OmegaConf.to_container here since it's a single value, not a complex structure, but keeping consistent with other config logging
            "split": OmegaConf.to_container(cfg.split, resolve=True),#OmegaConf.to_container converts DictConfig to regular dict for better logging and handling in wandb, resolve=True resolves any references in the config (like ${...}) to their actual values
            "cv": OmegaConf.to_container(cfg.cv, resolve=True),
            "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
            "model": run_cfg["model"],#different extraction due to how run config is created. Should include model architecture name and hyperparameters for current run
        },
        settings=wandb.Settings(
            init_timeout=cfg.logger.get("init_timeout", 180),
            x_disable_stats=True,
            x_disable_meta=True,
            x_disable_machine_info=True,
        ),
        #for multiple runs in one python process as is the case in sweeping, to make sure each run is treated as a separate run in wandb and not as a continuation of previous run
        reinit="finish_previous",
    )
    #returning wandb run object
    return run#currently not used, but leaving for structure and possible future use
#main 
@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    #importing code form modules (seprate files for better orgnization of logic)
    from data import load_data, train_test_split
    from train import train
    #data loading
    features,labels = load_data()
    #splitting data into train and test sets
    features_train, features_test, labels_train, labels_test = train_test_split()
    #getting config for each wandb run
    run_cfg = expand_search_space(cfg.search)
    #preparing variable to hold results
    sweep_results = []
    #going through all configurations
    for model_cfg in run_cfg:
        #starting wandb run for current configuration
        run = start_wandb_run(cfg, model_cfg, job_type="sweep")
        #training
        _ , results = train()
        #logging direct results of current run
        run.log(
            {
                "train_acc": results["train_acc"],
                "val_acc": results["val_acc"],
            }
        )
        #finishing current run after logging all needed info
        wandb.finish()
        #adding results of current run to program results (not wandb logging)
        sweep_results.append(
            {
                "model_cfg": deepcopy(model_cfg),
                "train_acc": results["train_acc"],
                "val_acc": results["val_acc"],
            }
        )
    ##-----best configuration is being retrained as it is not feasible to save all models during training and then just pick the best one for evaluation on test set, so we need to retrain the best model configuration on the whole training set and evaluate on test(validation) set (which is the same models were trained in the sweep)-----##
    #getting best configuration based on validation accuracy (the main metric for model selection) from all runs in sweep results
    best_result = max(sweep_results, key=lambda x: x["val_acc"])
    best_model_cfg = best_result["model_cfg"]
    #making run for final model training and evaluation with best configuration (based on validation accuracy) on test set and logging it in wandb
    start_wandb_run(cfg, best_model_cfg, job_type="final")
    #starting wandb run for current configuration
    run = start_wandb_run(cfg, model_cfg, job_type="sweep")
    #training
    model, results = train()
   #logging direct results of current run
    run.log(
        {
            "train_acc": results["train_acc"],
            "val_acc": results["val_acc"],
        }
    )
    #getting save paths for for current hydra run
    run_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "best_model.pt")
    metrics_path = os.path.join(run_dir, "best_metrics.json")
    #saving best model to hydra run
    torch.save(model.state_dict(), model_path)
    #saving metrics and configuration of the best model
    with open(metrics_path, "w") as file:#opens/creates json file for writing config and metrics of the best model
        json.dump(
            {
                "best_model_cfg": best_model_cfg,
                "train_acc": best_result["train_acc"],
                "val_acc": best_result["val_acc"],
            },
            file,
            indent=2,
        )
    #logging results of best models as artifact to wandb
    artifact = wandb.Artifact("best-model", type="model")
    artifact.add_file(model_path)
    artifact.add_file(metrics_path)
    wandb.log_artifact(artifact)
    #finishing run of best model
    wandb.finish()


if __name__ == "__main__":
    main()

