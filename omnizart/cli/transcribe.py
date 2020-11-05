# pylint: disable=C0303,W1401
import click

from omnizart.transcribe_all import process
from omnizart.utils import get_logger
from omnizart.cli.common_options import add_common_options, COMMON_TRANSCRIBE_OPTIONS


logger = get_logger("Trancribe CLI")


@click.command()
@add_common_options(COMMON_TRANSCRIBE_OPTIONS)
def transcribe(input_audio, model_path, output):
    """(Preparing) Transcribe all the information in the given audio.

    Supports to transcribe notes of instruments, drum percussion, chord progression,
    vocal melody, and beat position. Outputs the results as MIDI and CSV file.
    """
    process(input_audio=input_audio, model_path=model_path, output=output)
