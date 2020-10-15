import click


def add_common_options(options):
    def add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return add_options


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
        default="+"
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
