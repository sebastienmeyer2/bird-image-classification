"""Run optimized cross validation for parameters gridsearch."""


from typing import Any, Dict

from optuna.trial import Trial

from datasets.dataset_dict import DatasetDict


from engine.training import train_eval


class Objective():
    """An Objective class to wrap trials.

    General class that implements call functions for gridsearch algorithms.
    """
    def __init__(self, seed: int, model_name: str, dataset: DatasetDict):
        """The constructor of the class.

        Parameters
        ----------
        seed : int
            The seed to use everywhere for reproducibility.

        model_name : str
            The name of model following project usage. See README.md for more information.

        dataset : `DatasetDict`
            A dictionary containing training and validation sets.
        """
        # Handling randomness
        self.seed = seed

        # Data
        self.dataset = dataset

        # Model and decision values
        self.model_name = model_name

        # Model parameters
        self.params: Dict[str, Any] = {}

        # Keep best results in memory
        self.best_score = 0.

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for associated model.

        Returns
        -------
        params : dict of {str: any}
            A dictionary of parameters for chosen **model_name**. Initially, this attribute is
            empty and it is changed as many times as there are trials.
        """
        return self.params

    def set_params(self, params: Dict[str, Any]):
        """Set parameters for associated model.

        Parameters
        ----------
        params : dict of {str: any}
            A dictionary of parameters for chosen **model_name**. It contains all parameters to
            initialize and fit the model.
        """
        self.params = params

    def __call__(self, trial: Trial) -> float:
        """Run a trial using `optuna` package.

        Parameters
        ----------
        trial : Trial
            An instance of `Trial` object from `optuna` package to handle parameters search.

        Returns
        ----------
        trial_target : float
            Target metric value of current model during trial.
        """
        # Initialize parameter grid via optuna
        optuna_params = optuna_param_grid(trial, self.seed, self.model_name)

        self.set_params(optuna_params)

        # Run cross-validation
        trial_target = train_eval(self.model_name, self.params, self.dataset)

        # Keep best values in memory
        if trial_target > self.best_score:

            self.best_score = trial_target

        return trial_target


def optuna_param_grid(trial: Trial, seed: int, model_name: str) -> Dict[str, Any]:
    """Create a param grid for `optuna` usage.

    Parameters
    ----------
    trial : Trial
        An instance of `Trial` object from `optuna` package to handle parameters search.

    seed : int
        The seed to use everywhere for reproducibility.

    model_name : str
        The name of model following project usage. See README.md for more information.

    Returns
    -------
    params : dict of {str: any}
        A dictionary of parameters for chosen **model_name**. It contains all parameters to
        initialize and fit the model.

    Raises
    ------
    ValueError
        If no gridsearch corresponds to the **model_name**.
    """
    # Initialize model params
    params = {}

    if model_name in {"baseline", "evolved"}:

        # CNN

        # Main parameters
        params["random_state"] = trial.suggest_categorical("random_state", [seed])

        # Training parameters
        params["batch_size"] = trial.suggest_categorical("batch_size", [64])
        params["epochs"] = trial.suggest_categorical("epochs", [10])

        # Optimizer parameters
        params["optim_name"] = trial.suggest_categorical("optim_name", ["sgd"])
        params["learning_rate"] = trial.suggest_categorical("learning_rate", [0.01])
        params["momentum"] = trial.suggest_categorical("momentum", [0.5])

        # End of CNN

    elif model_name == "transfered":

        # Transfered CNN

        # Main parameters
        params["random_state"] = trial.suggest_categorical("random_state", [seed])

        # Training parameters
        params["batch_size"] = trial.suggest_categorical("batch_size", [64])
        params["epochs"] = trial.suggest_categorical("epochs", [10])

        # Optimizer parameters
        params["learning_rate"] = trial.suggest_categorical("learning_rate", [0.001])
        params["optim_name"] = trial.suggest_categorical("optim_name", ["adam"])
        if params["optim_name"] == "sgd":
            params["momentum"] = trial.suggest_float("momentum", 0., 0.95)
        elif params["optim_name"] == "adam":
            params["beta_1"] = trial.suggest_categorical("beta_1", [0.9])
            params["beta_2"] = trial.suggest_categorical("beta_2", [0.999])
        elif params["optim_name"] == "rmsprop":
            params["alpha"] = trial.suggest_float("alpha", 0.8, 0.95)
            params["momentum"] = trial.suggest_float("momentum", 0., 0.95)

        # End of Transfered CNN

    elif model_name == "google":

        # Transfered ViT from Google

        # Main parameters
        params["random_state"] = trial.suggest_categorical("random_state", [seed])

        # Training parameters
        params["batch_size"] = trial.suggest_categorical("batch_size", [12])
        params["epochs"] = trial.suggest_categorical("epochs", [20])

        # Optimizer parameters
        params["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 1e-3)
        params["optim_name"] = trial.suggest_categorical("optim_name", ["sgd"])
        if params["optim_name"] == "sgd":
            params["momentum"] = trial.suggest_float("momentum", 0.5, 0.99)
        elif params["optim_name"] == "adam":
            params["beta_1"] = trial.suggest_categorical("beta_1", [0.9])
            params["beta_2"] = trial.suggest_categorical("beta_2", [0.999])
        elif params["optim_name"] == "rmsprop":
            params["alpha"] = trial.suggest_float("alpha", 0.8, 0.95)
            params["momentum"] = trial.suggest_float("momentum", 0.4, 0.95)

        # Transforms
        params["min_scale"] = trial.suggest_float("min_scale", 0.3, 0.8)
        params["max_scale"] = trial.suggest_float("max_scale", 0.8, 1.0)
        params["min_ratio"] = trial.suggest_float("min_ratio", 0.6, 0.9)
        params["max_ratio"] = trial.suggest_float("max_ratio", 1.1, 1.5)

        params["do_jitter"] = trial.suggest_categorical("do_jitter", [False])
        if params["do_jitter"]:
            params["brightness"] = trial.suggest_float("brightness", 0.1, 0.25)
            params["contrast"] = trial.suggest_float("contrast", 0.1, 0.25)
            params["saturation"] = trial.suggest_float("saturation", 0.1, 0.25)
            params["hue"] = trial.suggest_float("hue", 0.0, 0.1)

        params["do_rotation"] = trial.suggest_categorical("do_rotation", [True, False])
        if params["do_rotation"]:
            params["degrees"] = trial.suggest_int("degrees", 10, 70)

        params["do_hflip"] = trial.suggest_categorical("do_hflip", [True, False])
        if params["do_hflip"]:
            params["p"] = trial.suggest_float("p", 0.25, 0.75)

        # End of Transfered ViT from Google

    else:

        err_msg = f"Unable to create optuna gridsearch for {model_name}."
        raise ValueError(err_msg)

    return params
