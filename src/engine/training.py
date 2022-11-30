"""Training and evaluation functions for general models."""


from typing import Any, Dict

from datasets.dataset_dict import DatasetDict


from engine.hub import create_model


def train_eval(model_name: str, params: Dict[str, Any], dataset: DatasetDict) -> float:
    """Fit a model on training set and compute metrics on evaluation set.

    The model is expected to possess the fit and predict methods.

    Parameters
    ----------
    model_name : str
        The name of model following project usage. See README.md for more information about
        available models.

    params : dict of {str: any}
        A dictionary of parameters for chosen **model_name**.

    dataset : `DatasetDict`
        A dictionary containing training and validation sets.

    Returns
    -------
    score : float
        **eval_metric** of chosen **model_name** and **params** on the fold.

    Raises
    ------
    ValueError
        If the **eval_metric** is unsupported.
    """
    # Initialize model
    model_raw = create_model(model_name, params)

    # Fit on whole training set
    score = model_raw.fit(dataset)

    return score
