from functools import partial

import click

click.option = partial(click.option, show_default=True)


@click.command()
@click.option(
    "-e",
    "--mode",
    help="Determine the evaluation mode.",
    type=click.Choice(["note", "note-stream", "frame", "frame-stream"]),
    default="note-stream",
)
@click.option(
    "-d",
    "--dataset-path",
    help="Path to the dataset, which should contain extracted testing feature.",
    type=click.Path(exists=True),
)
@click.option("-m", "--model-path", help="Path to the pre-trained model.", type=click.Path(exists=True))
@click.option("-s", "--pred-save-path", help="Path to save the prediction.", type=click.Path(writable=True))
@click.option("-p", "--pred-path", help="Path to the generated prediction.", type=click.Path(exists=True))
@click.option("--onset-th", help="Explicitly determine the onset threshold.", type=float, default=6)
def evaluate(
    dataset_path: str,
    model_path: str,
    pred_save_path: str,
    pred_path: str,
    mode: str = "note-stream",
    onset_th: float = 6.0,
):
    """Make prediction and evaluate on the dataset.

    To evaluate the performance of the model, one must generate the predictions first
    by assiging the flags '--feature-path', '--model-path', and '--pred-save-path'.

    To evaluate on the existing prediction, just assign the flag '--pred-path'. You
    can additionally assign the flag '--onset-th' to use different onset threshold.

    Different evaluation modes are also available. Available options are: ["note",
    "note-stream", "frame", "frame-stream"]. With postfix 'stream' means to evaluate
    on multiple instruments, and without it refers to evaluate on only pitch
    information.
    """
    pass
