import click

from omnizart.cli.common_options import add_common_options, COMMON_GEN_FEATURE_OPTIONS
from omnizart.setting_loaders import DrumSettings
from omnizart.utils import LazyLoader


drum = LazyLoader("drum", globals(), "omnizart.drum")


@click.command()
@add_common_options(COMMON_GEN_FEATURE_OPTIONS)
def generate_feature(dataset_path, output_path, num_threads):
    """Extract the feature of the whole dataset for training."""
    settings = DrumSettings()
    if output_path is not None:
        settings.dataset.feature_save_path = output_path

    drum.app.generate_feature(dataset_path, drum_settings=settings, num_threads=num_threads)
