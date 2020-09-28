from functools import partial

import click

from omnizart.music import app
from omnizart.setting_loaders import MusicSettings


click.option = partial(click.option, show_default=True)


@click.command()
@click.option(
    "-d",
    "--feature-path",
    help="Path to the folder of extracted feature",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "-m",
    "--model-name",
    help="Name for the output model (can be a path)",
    type=click.Path(writable=True)
)
@click.option(
    "-y",
    "--model-type",
    help="Type of the neural network model",
    type=click.Choice(["attn", "aspp"]),
    default="attn",
)
@click.option(
    "-i",
    "--input-model",
    help="If given, the training will continue to fine-tune on the pre-trained model.",
    type=click.Path(exists=True, writable=True),
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
@click.option("-e", "--epochs", help="Number of training epochs", type=int, default=20)
@click.option("-s", "--steps", help="Number of training steps of each epoch", type=int, default=3000)
@click.option("-vs", "--val-steps", help="Number of validation steps of each epoch", type=int, default=500)
@click.option("-b", "--batch-size", help="Batch size of each training step", type=int, default=16)
@click.option("-vb", "--val-batch-size", help="Batch size of each validation step", type=int, default=16)
@click.option(
    "--early-stop",
    help="Stop the training after the given epoch number if the validation accuracy did not improve.",
    type=int,
    default=6,
)
def train_model(
    feature_path,
    model_name,
    model_type,
    input_model,
    feature_type,
    label_type,
    loss_function,
    timesteps,
    epochs,
    steps,
    val_steps,
    batch_size,
    val_batch_size,
    early_stop,
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
