from functools import partial

import click

from omnizart.cli.common_options import add_common_options, COMMON_TRAIN_MODEL_OPTIONS
from omnizart.music import app
from omnizart.setting_loaders import MusicSettings


click.option = partial(click.option, show_default=True)


@click.command()
@add_common_options(COMMON_TRAIN_MODEL_OPTIONS)
@click.option(
    "-y",
    "--model-type",
    help="Type of the neural network model",
    type=click.Choice(["attn", "aspp"]),
    default="attn",
)
@click.option(
    "-f",
    "--feature-type",
    help="Determine the input feature types for training",
    multiple=True,
    default=["Spec", "Ceps"],
    type=click.Choice(["Spec", "Ceps", "GCoS"]),
)
@click.option(
    "-l",
    "--label-type",
    help="Determine the output label should be note- (onset, duration) or stream-level (onset, duration, instrument)",
    type=click.Choice(["note", "note-stream", "pop-note-stream", "frame", "frame-stream"]),
    default="note-stream",
)
@click.option(
    "-s",
    "--loss-function",
    help="Detemine which loss function to use",
    type=click.Choice(["focal", "smooth", "bce"]),
    default="smooth",
)
@click.option("-t", "--timesteps", help="Time width of each input feature", type=int, default=256)
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
    model_type,
    feature_type,
    label_type,
    loss_function,
    timesteps
):
    """Train a new model or continue to train on a pre-trained model"""
    settings = MusicSettings()
    settings.training.channels = feature_type
    settings.training.label_type = label_type
    settings.training.loss_function = loss_function
    settings.training.timesteps = timesteps
    settings.training.epoch = epochs
    settings.training.steps = steps
    settings.training.val_steps = val_steps
    settings.training.batch_size = batch_size
    settings.training.val_batch_size = val_batch_size
    settings.training.early_stop = early_stop
    settings.model.model_type = model_type

    app.train(feature_path, model_name=model_name, input_model_path=input_model, music_settings=settings)
