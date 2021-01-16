import click

from omnizart.cli.common_options import add_common_options, COMMON_GEN_FEATURE_OPTIONS
from omnizart.setting_loaders import BeatSettings
from omnizart.utils import LazyLoader


beat = LazyLoader("beat", globals(), "omnizart.beat")


@click.command()
@add_common_options(COMMON_GEN_FEATURE_OPTIONS)
def generate_feature(dataset_path, output_path, num_threads):
    """Extract the feature of the whole dataset for training."""
    settings = BeatSettings()

    if output_path is not None:
        settings.dataset.feature_save_path = output_path

    beat.app.generate_feature(dataset_path, beat_settings=settings, num_threads=num_threads)
