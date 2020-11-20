from functools import partial

import click

from omnizart.cli.common_options import add_common_options, COMMON_TRAIN_MODEL_OPTIONS
from omnizart.vocal_frame import app
from omnizart.setting_loaders import VocalFrameSettings


click.option = partial(click.option, show_default=True)


@click.command()
@add_common_options(COMMON_TRAIN_MODEL_OPTIONS)
@click.option("-t", "--timesteps", help="Time width of each input feature", type=int, default=128)
def train_model(
    feature_path,
    model_name,
    input_model,
    epochs,
    steps,
    batch_size,
    early_stop,
    timesteps
):
    """Train a new model or continue to train on a pre-trained model"""
    settings = VocalFrameSettings()
    settings.training.timesteps = timesteps
    if epochs is not None:
        settings.training.epoch = epochs
    if steps is not None:
        settings.training.steps = steps
    if batch_size is not None:
        settings.training.batch_size = batch_size
    if early_stop is not None:
        settings.training.early_stop = early_stop

    app.train(feature_path, model_name=model_name, input_model_path=input_model, music_settings=settings)
