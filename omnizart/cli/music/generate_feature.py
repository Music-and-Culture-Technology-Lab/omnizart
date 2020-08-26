import click


@click.command()
@click.option(
    "-d", "--dataset-path", help="Path to the downloaded dataset", type=click.Path(exists=True), required=True
)
@click.option("-h", "--harmonic", help="Wether to use harmonic version of the feature", is_flag=True)
def generate_feature(dataset, harmonic):
    """Extract the feature of the whole dataset for future use.

    Available datasets are:
        Maps: Piano solo performances (smaller)
        Maestro: Piano solo performances (larger)
        MusicNet: Classical music performances, with 11 classes of instruments
        Rhythm: Pop music, including various instruments, drums, and vocal.
    """
    pass
