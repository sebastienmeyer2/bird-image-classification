"""Main file to run trials and test models."""


import os

from typing import List

from datetime import datetime

import argparse

from tqdm import tqdm

import numpy as np

import optuna

from datasets import load_dataset


from engine.gridsearch import Objective
from engine.hub import create_model
from utils.data_preparation import pil_loader
from utils.seed_handler import SeedHandler


# pylint: disable=dangerous-default-value
def run(
    seed: int = 42,
    models_names: List[str] = ["baseline"],
    data_path: str = "data",
    n_trials: int = 25
):
    """Run the gridseach and eventually submit predictions.

    Parameters
    ----------
    seed : int, default=42
        The seed to use everywhere for reproducibility.

    models_names : list of str, default=["baseline"]
        Name of the models following project usage. See README.md for more information.

    data_path : str, default="data"
        Path to the directory where the data is stored.

    n_trials : int, default=25
        Number of trials for `optuna` gridsearch.
    """
    # Fix seed
    sh = SeedHandler()

    sh.set_seed(seed)
    sh.init_seed()

    # Get data
    dataset = load_dataset("imagefolder", data_dir=data_path)

    for model_name in models_names:

        # Research grid parameters
        study_id = datetime.now().strftime("%d-%m-%y_%H-%M-%S")

        # Create Objective
        objective = Objective(seed, model_name, dataset)

        # Initialize optuna study object
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(study_name=study_id, direction="maximize", sampler=sampler)

        # Run whole gridsearch
        study.optimize(objective, n_trials=n_trials)

        if n_trials > 0:
            best_params = study.best_trial.params
        else:
            best_params = {"random_state": seed}

        # Print summary
        summary: List[str] = []

        summary.append("\n===== OPTUNA GRID SEARCH SUMMARY =====\n")
        summary.append(f"Model is {model_name}.\n")
        summary.append("\n")
        summary.append(f"Cross-validation training accuracy: {objective.best_score}.\n")
        summary.append("\n")
        summary.append(f"Current params are:\n {best_params}\n")
        summary.append("=========================================\n")

        print("".join(summary))

        # Save submission
        best_model = create_model(model_name, best_params)
        best_model.fit(dataset)

        test_dir = data_path + "/test_images/mistery_category"

        # Save file in the requested format
        filename = f"results/submission_{study_id}"
        filename += f"_acc_{np.around(float(objective.best_score), 3)}.csv"
        output_file = open(filename, "w", encoding="utf-8")
        output_file.write("Id,Category\n")

        transform = best_model.create_transform()

        for f in tqdm(os.listdir(test_dir)):

            if "jpg" in f:

                data = transform(pil_loader(f"{test_dir}/{f}"))
                data = data.view(1, data.size(0), data.size(1), data.size(2))
                use_cuda = True
                if use_cuda:
                    data = data.cuda()
                output = best_model.model(data)
                if isinstance(output, dict):
                    output = output.logits
                pred = output.data.max(1, keepdim=True)[1]
                output_file.write("%s,%d\n" % (f[:-4], pred))

        output_file.close()

        print(f"=====> A submission file has been created under {filename}.")
# pylint: enable=dangerous-default-value


if __name__ == "__main__":

    # Command lines
    PARSER_DESC = "Main file to train a model and make predictions."
    PARSER = argparse.ArgumentParser(description=PARSER_DESC)

    # Seed
    PARSER.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed to use everywhere for reproducbility. Default: 42."
    )

    # Model name
    PARSER.add_argument(
        "--models-names",
        default=["baseline"],
        nargs="*",
        help="""
             Choose models names. Available models: "baseline", "evolved", "transfered" and
             "google".
             """
    )

    # Path to data
    PARSER.add_argument(
        "--data-path",
        default="data/",
        type=str,
        help="""
             Path to the directory where the data is stored. Default: "data/".
             """
    )

    # Number of trials
    PARSER.add_argument(
        "--trials",
        default=25,
        type=int,
        help="Choose the number of gridsearch trials. Default: 25."
    )

    # End of command lines
    ARGS = PARSER.parse_args()

    # Run the gridsearch
    run(
        seed=ARGS.seed,
        models_names=ARGS.models_names,
        data_path=ARGS.data_path,
        n_trials=ARGS.trials,
    )
