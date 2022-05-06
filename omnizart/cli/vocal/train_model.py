import click

from omnizart.cli import silence_tensorflow
from omnizart.cli.common_options import (COMMON_TRAIN_MODEL_OPTIONS,
                                         add_common_options)
from omnizart.setting_loaders import VocalSettings
from omnizart.utils import LazyLoader

vocal = LazyLoader("vocal", globals(), "omnizart.vocal")


@click.command()
@add_common_options(COMMON_TRAIN_MODEL_OPTIONS)
def train_model(
    feature_path,
    model_name,
    input_model,
    epochs,
    steps,
    val_steps,
    batch_size,
    val_batch_size,
    early_stop,
):
    """Train a new model or continue to train on a pre-trained model"""
    settings = VocalSettings()

    if epochs is not None:
        settings.training.epoch = epochs
    if steps is not None:
        settings.training.steps = steps
    if batch_size is not None:
        settings.training.batch_size = batch_size
    if val_steps is not None:
        settings.training.val_steps = val_steps
    if val_batch_size is not None:
        settings.training.val_batch_size = val_batch_size
    if early_stop is not None:
        settings.training.early_stop = early_stop

    silence_tensorflow()
    vocal.app.train(feature_path, model_name=model_name, input_model_path=input_model, vocal_settings=settings)
