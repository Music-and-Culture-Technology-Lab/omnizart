import click

from omnizart.cli.common_options import add_common_options, COMMON_GEN_FEATURE_OPTIONS
from omnizart.setting_loaders import PatchCNNSettings
from omnizart.utils import LazyLoader


patch_cnn = LazyLoader("patch_cnn", globals(), "omnizart.patch_cnn")


@click.command()
@add_common_options(COMMON_GEN_FEATURE_OPTIONS)
def generate_feature(dataset_path, output_path, num_threads):
    """Pre-process the dataset for training.

    \b
    Supported datasets are:
    * MIR-1K
    """
    settings = PatchCNNSettings()
    if output_path is not None:
        settings.dataset.feature_save_path = output_path

    patch_cnn.app.generate_feature(dataset_path, settings, num_threads=num_threads)
