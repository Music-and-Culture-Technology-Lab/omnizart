import click

from omnizart.cli.common_options import add_common_options, COMMON_GEN_FEATURE_OPTIONS
from omnizart.setting_loaders import VocalSettings
from omnizart.utils import LazyLoader


vocal = LazyLoader("vocal", globals(), "omnizart.vocal")


@click.command()
@add_common_options(COMMON_GEN_FEATURE_OPTIONS)
def generate_feature(dataset_path, output_path, num_threads):
    """Extract the feature of the whole dataset for training."""
    settings = VocalSettings()
    if output_path is not None:
        settings.dataset.feature_save_path = output_path

    vocal.app.generate_feature(dataset_path, vocal_settings=settings, num_threads=num_threads)
