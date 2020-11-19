import click

from omnizart.cli.common_options import add_common_options, COMMON_GEN_FEATURE_OPTIONS
from omnizart.setting_loaders import MusicSettings
from omnizart.utils import LazyLoader


music = LazyLoader("music", globals(), "omnizart.music")


@click.command()
@add_common_options(COMMON_GEN_FEATURE_OPTIONS)
@click.option("-h", "--harmonic", help="Whether to use harmonic version of the feature", is_flag=True)
def generate_feature(dataset_path, output_path, num_threads, harmonic):
    """Extract the feature of the whole dataset for training.

    The command will try to infer the dataset type from the given dataset path.

    \b
    Available datasets are:
    * Maps: Piano solo performances (smaller)
    * Maestro: Piano solo performances (larger)
    * MusicNet: Classical music performances, with 11 classes of instruments
    * Pop: Pop music, including various instruments, drums, and vocal.
    """
    settings = MusicSettings()
    settings.feature.harmonic = harmonic
    if output_path is not None:
        settings.dataset.feature_save_path = output_path

    music.app.generate_feature(dataset_path, settings, num_threads=num_threads)
