"""The entry point of the ``omnizart`` command.

Sub-commands are also added here.

Examples
--------
.. code-block:: bash

    omnizart --help
    omnizart music --help
"""
import os

import click

import omnizart.constants.datasets as dset
from omnizart import MODULE_PATH
from omnizart.remote import download, download_large_file_from_google_drive
from omnizart.utils import ensure_path_exists
from omnizart.cli.music import music
from omnizart.cli.drum import drum
from omnizart.cli.chord import chord
from omnizart.cli.transcribe import transcribe


@click.group()
def entry():
    pass


@click.command()
@click.argument(
    "dataset", type=click.Choice(["Maestro", "MusicNet", "McGill", "BPS-FH", "Ext-Su"], case_sensitive=False)
)
@click.option(
    "-o", "--output", default="./", help="Path for saving the downloaded dataset.", type=click.Path(writable=True)
)
@click.option("--unzip", help="Whether to unzip the downloaded dataset", is_flag=True)
def download_dataset(dataset, output, unzip):
    """A quick command for downloading datasets."""
    url = {
        "maestro": dset.MaestroStructure.url,
        "musicnet": dset.MusicNetStructure.url,
        "mcgill": dset.McGillBillBoard.url,
        "bps-fh": dset.BeethovenSonatas.url,
        "ext-su": dset.ExtSuStructure.url
    }[dataset.lower()]
    ensure_path_exists(output)
    click.echo(f"Downloading {dataset} dataset and save to {output}")
    if "drive.google.com" in url:
        download_large_file_from_google_drive(url, save_path=output, save_name=dataset + ".zip", unzip=unzip)
    else:
        download(url, save_path=output, unzip=unzip)


@click.command()
def download_checkpoints():
    """Downlaod the archived checkpoints of different models."""
    CHECKPOINTS = {
        "chord_v1": {
            "fid": "1QX5bBoYzZyC2fvK26YEtF_Hqt3DzhiHk",
            "save_as": "checkpoints/chord/chord_v1/weights.data-00000-of-00001",
            "file_length": 132717707
        },
        "drum_keras": {
            "fid": "1seqz_pi20zB8rq1YJE0Jbk1SwkJ9hOCK",
            "save_as": "checkpoints/drum/drum_keras/weights.h5",
            "file_length": 31204608
        },
        "music_pop": {
            "fid": "1-kM27jR_iCvF8Z-3pAFG-nrMRktyxxJ0",
            "save_as": "checkpoints/music/music_pop/weights.h5",
            "file_length": 31892440
        },
        "music_piano": {
            "fid": "1x9_qjXSiM4GAxpvKfdYJK5S3SLdlCl2I",
            "save_as": "checkpoints/music/music_piano/weights.h5",
            "file_length": 50738464
        }
    }

    for checkpoint, info in CHECKPOINTS.items():
        print(f"Downloading checkpoints: {checkpoint}")
        save_name = os.path.basename(info["save_as"])
        save_path = os.path.dirname(info["save_as"])
        save_path = os.path.join(MODULE_PATH, save_path)
        download_large_file_from_google_drive(
            info["fid"], file_length=info["file_length"], save_path=save_path, save_name=save_name
        )


entry.add_command(music)
entry.add_command(drum)
entry.add_command(chord)
entry.add_command(transcribe)
entry.add_command(download_dataset)
entry.add_command(download_checkpoints)
