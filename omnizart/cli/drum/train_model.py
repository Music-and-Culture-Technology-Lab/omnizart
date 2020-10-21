import click

from omnizart.cli.common_options import add_common_options, COMMON_TRAIN_MODEL_OPTIONS
from omnizart.drum import app
from omnizart.setting_loaders import DrumSettings


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
    settings = DrumSettings()
    settings.training.epoch = epochs
    settings.training.steps = steps
    settings.training.batch_size = batch_size
    settings.training.val_steps = val_steps
    settings.training.val_batch_size = val_batch_size
    settings.training.early_stop = early_stop

    app.train(feature_path, model_name=model_name, input_model_path=input_model, drum_settings=settings)
