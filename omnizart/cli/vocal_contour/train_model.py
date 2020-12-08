import click

from omnizart.cli.common_options import add_common_options, COMMON_TRAIN_MODEL_OPTIONS
from omnizart.setting_loaders import VocalContourSettings
from omnizart.utils import LazyLoader


vocal_contour = LazyLoader("vocal_contour", globals(), "omnizart.vocal_contour")


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
    early_stop
):
    """Train a new model or continue to train on a pre-trained model"""
    settings = VocalContourSettings()
    if epochs is not None:
        settings.training.epoch = epochs
    if steps is not None:
        settings.training.steps = steps
    if val_steps is not None:
        settings.training.val_steps = val_steps
    if batch_size is not None:
        settings.training.batch_size = batch_size
    if val_batch_size is not None:
        settings.training.val_batch_size = val_batch_size
    if early_stop is not None:
        settings.training.early_stop = early_stop

    vocal_contour.app.train(
        feature_path, model_name=model_name, input_model_path=input_model, vocalcontour_settings=settings
    )
