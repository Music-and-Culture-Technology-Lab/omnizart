import click

from omnizart.cli import silence_tensorflow
from omnizart.cli.common_options import add_common_options, COMMON_TRANSCRIBE_OPTIONS
from omnizart.utils import LazyLoader


patch_cnn = LazyLoader("patch_cnn", globals(), "omnizart.patch_cnn")


@click.command()
@add_common_options(COMMON_TRANSCRIBE_OPTIONS)
def transcribe(input_audio, model_path, output):
    """Transcribe a single audio and output CSV and audio file.

    The transcribed F0 contour will be stored in the <filename>_f0.csv file,
    where *filename* is the input file name. Also there will be another rendered
    audio file (with postfix <filename>_trans.wav) of the pitch contour for
    quick validation.

    Supported modes are: Melody
    """
    silence_tensorflow()
    patch_cnn.app.transcribe(input_audio, model_path, output=output)
