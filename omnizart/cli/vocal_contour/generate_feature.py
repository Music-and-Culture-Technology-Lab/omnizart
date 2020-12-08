from functools import partial
import click

from omnizart.cli.common_options import add_common_options, COMMON_GEN_FEATURE_OPTIONS
from omnizart.setting_loaders import VocalContourSettings
from omnizart.utils import LazyLoader


vocal_contour = LazyLoader("vocal_contour", globals(), "omnizart.vocal_contour")
click.option = partial(click.option, show_default=True)


@click.command()
@add_common_options(COMMON_GEN_FEATURE_OPTIONS)
@click.option(
    "-h",
    "--hop-size",
    help="Hop size in seconds with respect to sampling rate.",
    type=float,
    default=0.02
)
@click.option(
    "-s",
    "--sampling-rate",
    help="Adjust input sampling rate to this value.",
    type=int,
    default=16000
)
def generate_feature(dataset_path, output_path, num_threads, hop_size, sampling_rate):
    """Extract the feature of the whole dataset for training.

    The command will try to infer the dataset type from the given dataset path.

    \b
    * MIR-1K
    * MedleyDB
    """
    settings = VocalContourSettings()
    settings.feature.hop_size = hop_size
    settings.feature.sampling_rate = sampling_rate
    if output_path is not None:
        settings.dataset.feature_save_path = output_path

    vocal_contour.app.generate_feature(dataset_path, settings, num_threads=num_threads)
