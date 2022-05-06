# pylint: disable=C0303,W1401
import click

from omnizart.cli.common_options import (COMMON_TRANSCRIBE_OPTIONS,
                                         add_common_options)
from omnizart.utils import LazyLoader

vocal_contour = LazyLoader("vocal_contour", globals(), "omnizart.vocal_contour")


@click.command()
@add_common_options(COMMON_TRANSCRIBE_OPTIONS)
def transcribe(input_audio, model_path, output):
    """Transcribe a single audio and output as a WAV file.

    This will output a WAV file with the same name as the given audio, except the
    extension will be replaced with '.wav'.

    \b
    Example Usage
    $ omnizart vocal-contour transcribe \ 
        example.wav \ 
        --model-path path/to/model \ 
        --output example.mid
    """
    vocal_contour.app.transcribe(input_audio, model_path, output=output)


def process_doc():
    # Some dirty work for preserving and converting the docstring inside the decorated
    # function into .rst format.
    doc = transcribe.__doc__
    doc = doc.replace("\b", "").replace("    ", "").replace("--", "        --")

    code_block = "\n.. code-block:: bash\n\n"
    doc = doc.replace("$", f"{code_block}    $")

    return doc


__doc__ = process_doc()
