# pylint: disable=C0303,W1401

from functools import partial

import click

from omnizart.transcribe_all import process
from omnizart.cli import apps
from omnizart.utils import get_logger

logger = get_logger("Trancribe CLI")
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

click.option = partial(click.option, show_default=True)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("trans_type", type=click.Choice(list(apps.keys()) + ["all"], case_sensitive=False))
@click.argument("input_audio", type=click.Path(exists=True))
@click.option(
    "-m",
    "--model-path",
    help="Path to the pre-trained model for transcription",
    type=click.Path(exists=True),
)
@click.option("-o", "--output", help="Path to output the MIDI file", default="./", type=click.Path(writable=True))
def transcribe(trans_type, input_audio, model_path, output):
    """Collections of transcription commands.

    Transcribes the audio in different aspects, such as instrument, drum, and vocal. Specify `all` to the first
    argument to transcribe all different aspects. The output will be a MIDI file with the same name as the
    input audio, except the extension will be replaced with '.mid'.

    \b
    Available functions are: 
    * music - trancribes instrument notes
    * drum - transcribes drum percussions
    * vocal - transcribes vocal melodies
    * beat - MIDI-domain beat tracking
    * chord - MIDI-domain chord transcription
    Input for the former three should be wav file, and the last two
    should be MIDI file.

    \b
    Example Usage
    $ omnizart transcribe music \ 
        example.wav \ 
        --model-path path/to/model \ 
        --output example.mid
    """
    if trans_type == "all":
        process(apps, input_audio=input_audio, model_path=model_path)
    else:
        apps[trans_type.lower()].transcribe(input_audio, model_path, output=output)


def process_doc():
    # Some dirty work for preserving and converting the docstring inside the decorated
    # function into .rst format.
    doc = transcribe.__doc__
    secs = doc.split("\n\n")
    secs = [sec.replace("    ", "") for sec in secs]

    # List of available functions
    func_sec = secs[2]
    func_sec = func_sec.replace("functions are:", "functions are:\n\n")
    func_sec = func_sec.replace("* ", "* ``").replace(" -", "`` -")
    secs[2] = func_sec

    # Example section
    example_sec = secs[-1]
    space = " " * 8
    example_sec = (
        example_sec.replace("\b", "").replace("--", space + "--").replace("example.wav", space + "example.wav")
    )
    code_block = "\n.. code-block:: bash\n\n"
    example_sec = example_sec.replace("$", f"{code_block}    $")

    secs[-1] = example_sec
    doc = "\n\n".join(secs)

    return doc


__doc__ = process_doc()
