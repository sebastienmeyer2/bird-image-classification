"""Initialize a model based on its name and parameters."""


from typing import Any, Dict, Union


from engine.models.cnn import CNNEstimator
from engine.models.transformer import TransformerEstimator


CNN_MODELS_NAMES = [
    "baseline", "evolved", "transfered"
]

TRANSFORMER_MODELS_NAMES = [
    "google"
]


def create_model(
    model_name: str, params: Dict[str, Any]
) -> Union[CNNEstimator, TransformerEstimator]:
    """Create a model.

    Parameters
    ----------
    model_name : str
        The name of model following project usage. See README.md for more information about
        available models.

    params : dict of str
        A dictionary of parameters for chosen **model_name**.

    Returns
    -------
    model : `CNNEstimator` or `TransformerEstimator`
        Corresponding model from the catalogue.

    Raises
    ------
    ValueError
        If the **model_name** is not supported.
    """
    if model_name in CNN_MODELS_NAMES:

        model = CNNEstimator(model_name, params)

    elif model_name in TRANSFORMER_MODELS_NAMES:

        model = TransformerEstimator(model_name, params)

    else:

        err_msg = f"Unknown model {model_name}."
        raise ValueError(err_msg)

    return model
