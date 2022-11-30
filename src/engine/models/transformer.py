"""Wrapper for Transformer models."""


from functools import partial

from typing import Any, Dict, Union

import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, RMSprop, Adam
from torch.utils.data import DataLoader
from torchvision.transforms import (
    ColorJitter, Compose, InterpolationMode, Normalize, RandomHorizontalFlip, RandomResizedCrop,
    RandomRotation, ToTensor
)

from datasets.dataset_dict import DatasetDict
from transformers import AutoModelForImageClassification, get_scheduler, AutoFeatureExtractor


from utils.data_preparation import apply_transform, collate_fn


class TransformerEstimator():
    """Transfer learning of Google ViT."""

    def __init__(self, model_name: str, params: Dict[str, Any], n_classes: int = 20):
        """Initialize the model.

        Parameters
        ----------
        model_name : str
            The name of model following project usage. See README.md for more information. Only
            present for compatibility.

        params : dict of {str: any}
            A dictionary of parameters for chosen **model_name**. It contains all parameters to
            initialize and fit the model.

        n_classes : int, default=20
            Number of classes.
        """
        self.model_name = model_name

        # Model parameters
        self.model_params = params

        # Base architecture
        self.model_checkpoint = "google/vit-base-patch16-224"

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_checkpoint, num_labels=n_classes, ignore_mismatched_sizes=True
        )

        # Move to CUDA
        self.model.cuda()

    def fit(self, dataset: DatasetDict) -> float:
        """Wrapper of fit method.

        Parameters
        ----------
        dataset : `DatasetDict`
            A dictionary containing training and validation sets.

        Returns
        -------
        best_val_acc : float
            Best obtained validation accuracy for all epochs until early stopping or end.
        """
        # Get parameters
        batch_size = self.model_params.get("batch_size", 12)
        epochs = self.model_params.get("epochs", 20)
        verbose = self.model_params.get("verbose", 1)
        patience = self.model_params.get("patience", 2)

        # Apply transforms on the fly
        transforms = self.create_transform()

        dataset.set_transform(partial(apply_transform, transforms))

        train_set = dataset["train"]
        val_set = dataset["validation"]

        # Load dataset
        train_loader = DataLoader(
            train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=1
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=1
        )

        n_train = len(train_loader.dataset)
        n_train_batches = len(train_loader)

        n_val = len(val_loader.dataset)
        n_val_batches = len(val_loader)

        # Initialize optimizer and learning rate scheduler
        optimizer = self.create_optimizer()

        n_train_steps = epochs * n_train_batches
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0,
            num_training_steps=n_train_steps
        )

        # Start training
        use_cuda = True
        log_interval = 1

        val_loss_list = []
        val_acc_list = []

        for epoch in range(epochs):

            # Training
            self.model.train()

            train_loss = 0
            train_correct = 0

            for batch_idx, batch in enumerate(train_loader):

                if use_cuda:
                    batch = {k: v.cuda() for k, v in batch.items()}

                data = batch["pixel_values"]
                target = batch["labels"]

                # Set optimizer to zero grad
                optimizer.zero_grad()

                # Forward pass
                output = self.model(**batch).logits

                criterion = CrossEntropyLoss(reduction="mean")
                loss = criterion(output, target)

                # Backward pass
                loss.backward()

                optimizer.step()
                lr_scheduler.step()

                # Compute metrics and eventually print
                if verbose > 0 and batch_idx % log_interval == 0:

                    n_cur = batch_idx * len(data)
                    perc_cur = 100. * batch_idx / n_train_batches
                    log_msg = f"[{n_cur}/{n_train} ({perc_cur:.0f}%)]"

                    batch_loss = loss.data.item()
                    log_msg += f"\tBatch loss: {batch_loss:.3f}"

                    pred = output.data.max(1, keepdim=True)[1]
                    correct = pred.eq(target.data.view_as(pred)).cpu().sum()
                    batch_acc = 100. * correct / len(data)
                    log_msg += f"\tBatch acc: {batch_acc:.2f}"

                    print(log_msg)

                    train_loss += batch_loss  # total loss for epoch
                    train_correct += correct  # total number of correct predictions for epoch

            train_loss /= n_train_batches
            train_acc = 100. * train_correct / n_train

            log_msg = f"\nEpoch {epoch}; Training set: "
            log_msg += f"Average loss: {train_loss:.4f}; "
            log_msg += f"Accuracy: {train_correct}/{n_train} ({train_acc:.2f}%)"
            print(log_msg)

            # Validation
            self.model.eval()

            val_loss = 0
            val_correct = 0

            for batch in val_loader:

                if use_cuda:
                    batch = {k: v.cuda() for k, v in batch.items()}

                data = batch["pixel_values"]
                target = batch["labels"]

                # Forward pass
                output = self.model(**batch).logits

                criterion = CrossEntropyLoss(reduction="mean")
                val_loss += criterion(output, target).data.item()

                pred = output.data.max(1, keepdim=True)[1]
                val_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            val_loss /= n_val_batches
            val_acc = 100. * val_correct / n_val

            log_msg = "Validation set: "
            log_msg += f"Average loss: {val_loss:.4f}; "
            log_msg += f"Accuracy: {val_correct}/{n_val} ({val_acc:.2f}%)\n"
            print(log_msg)

            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            # Early stopping
            if len(val_acc_list) > patience:

                worse_acc = True
                for past_acc in val_acc_list[-1-patience:-1]:
                    if val_acc_list[-1] > past_acc:
                        worse_acc = False

                if worse_acc:

                    log_msg = "Early stopping - patience"
                    print(log_msg)

                    best_val_acc = np.max(val_acc_list)

                    return best_val_acc

            if val_acc < 10:

                log_msg = "Early stopping - very bad model"
                print(log_msg)

                best_val_acc = np.max(val_acc_list)

                return best_val_acc

        model_file = f"experiment/{self.model_name}_{str(epoch)}.pth"
        torch.save(self.model.state_dict(), model_file)

        log_msg = f"Saved model to {model_file}\n"
        print(log_msg)

        best_val_acc = np.max(val_acc_list)

        return best_val_acc

    def create_optimizer(self) -> Union[SGD, RMSprop, Adam]:
        """Build up an optimizer.

        Parameters
        ----------
        params : dict of {str: any}
            A dictionary of parameters for the model. It contains all parameters to initialize and
            fit the model.

        Returns
        -------
        optimizer : `SGD`, `Adam` or `RMSProp`
            Initialized optimizer.

        Raises
        ------
        ValueError
            If the optimizer is not supported.
        """
        optim_name = self.model_params.get("optim_name", "sgd")

        if optim_name == "sgd":

            learning_rate = self.model_params.get("learning_rate", 0.000668)
            momentum = self.model_params.get("momentum", 0.629)

            optimizer = SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)

        elif optim_name == "rmsprop":

            learning_rate = self.model_params.get("learning_rate", 0.01)
            alpha = self.model_params.get("alpha", 0.99)
            momentum = self.model_params.get("momentum", 0.)

            optimizer = RMSprop(
                self.model.parameters(), lr=learning_rate, alpha=alpha, momentum=momentum
            )

        elif optim_name == "adam":

            learning_rate = self.model_params.get("learning_rate", 0.001)
            beta_1 = self.model_params.get("beta_1", 0.9)
            beta_2 = self.model_params.get("beta_2", 0.999)

            optimizer = Adam(self.model.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

        else:

            err_msg = f"Unknown optimizer {optim_name}."
            raise ValueError(err_msg)

        return optimizer

    def create_transform(self) -> Compose:
        """Initialize transform to apply to all images.

        Returns
        -------
        transform : torchvision.transforms.Compose
            A sequence of transformations to apply to input images.
        """
        # Get parameters
        min_scale = self.model_params.get("min_scale", 0.523)
        max_scale = self.model_params.get("max_scale", 0.957)
        min_ratio = self.model_params.get("min_ratio", 0.704)
        max_ratio = self.model_params.get("max_ratio", 1.362)

        do_jitter = self.model_params.get("do_jitter", False)
        brightness = self.model_params.get("brightness", 0)
        contrast = self.model_params.get("contrast", 0)
        saturation = self.model_params.get("saturation", 0)
        hue = self.model_params.get("hue", 0)

        do_rotation = self.model_params.get("do_rotation", False)
        degrees = self.model_params.get("degrees", 10)

        do_hflip = self.model_params.get("do_hflip", False)
        p = self.model_params.get("p", 0.55)

        # List of transforms to apply
        transform_list = []

        if self.feature_extractor.do_resize:
            transform_list.append(
                RandomResizedCrop(
                    (self.feature_extractor.size, self.feature_extractor.size),
                    scale=(min_scale, max_scale), ratio=(min_ratio, max_ratio)
                )
            )

        if do_jitter:
            transform_list.append(
                ColorJitter(
                    brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
                )
            )

        if do_rotation:
            transform_list.append(RandomRotation(degrees, interpolation=InterpolationMode.BILINEAR))

        if do_hflip:
            transform_list.append(RandomHorizontalFlip(p=p))

        transform_list.append(ToTensor())

        if self.feature_extractor.do_normalize:
            transform_list.append(
                Normalize(
                    mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std
                )
            )

        transform = Compose(transform_list)

        return transform
