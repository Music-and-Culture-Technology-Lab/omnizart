import click

from omnizart.cli import silence_tensorflow
from omnizart.cli.common_options import add_common_options, COMMON_TRAIN_MODEL_OPTIONS
from omnizart.setting_loaders import ChordSettings
from omnizart.utils import LazyLoader


chord = LazyLoader("chord", globals(), "omnizart.chord")


@click.command()
@add_common_options(COMMON_TRAIN_MODEL_OPTIONS)
@click.option("--learning-rate-decay", help="Decaying rate of learning rate per epoch", type=int)
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
    learning_rate_decay
):
    """Train a new model or continue to train on a pre-trained model"""
    settings = ChordSettings()

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
    if learning_rate_decay is not None:
        settings.training.learning_rate_decay = learning_rate_decay

    silence_tensorflow()
    chord.app.train(feature_path, model_name=model_name, input_model_path=input_model, chord_settings=settings)
