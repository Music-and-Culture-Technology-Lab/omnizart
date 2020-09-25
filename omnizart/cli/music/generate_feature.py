import click

from omnizart.music import app
from omnizart.setting_loaders import MusicSettings


@click.command()
@click.option(
    "-d", "--dataset-path", help="Path to the downloaded dataset", type=click.Path(exists=True), required=True
)
@click.option(
    "-o",
    "--output-path",
    help="Path for saving the extracted feature. Default to the same folder of the dataset.",
    type=click.Path(writable=True),
    default="+"
)
@click.option("-h", "--harmonic", help="Whether to use harmonic version of the feature", is_flag=True)
def generate_feature(dataset_path, output_path, harmonic):
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
    settings.dataset.feature_save_path = output_path
    app.generate_feature(dataset_path, settings)
