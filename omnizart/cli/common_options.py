import click


def add_common_options(options):
    def add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return add_options


COMMON_TRANSCRIBE_OPTIONS = [
    click.argument("input_audio", type=click.Path(exists=True)),
    click.option(
        "-m",
        "--model-path",
        help="Path to the pre-trained model for transcription",
        type=click.Path(exists=True),
    ),
    click.option(
        "-o",
        "--output",
        help="Path to output the prediction file (could be MIDI, CSV, ..., etc.)",
        default="./",
        show_default=True,
        type=click.Path(writable=True)
    )
]


COMMON_GEN_FEATURE_OPTIONS = [
    click.option(
        "-d",
        "--dataset-path",
        help="Path to the downloaded dataset",
        type=click.Path(exists=True),
        required=True
    ),
    click.option(
        "-o",
        "--output-path",
        help="Path for svaing the extracted feature. Default to the folder under the dataset.",
        type=click.Path(writable=True),
        default="+",
        show_default=True,
    ),
    click.option(
        "-n",
        "--num-threads",
        help="Parallel extract the feature by using multiple threads.",
        type=int,
        default=4,
        show_default=True
    )
]
