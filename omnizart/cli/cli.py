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
from omnizart.utils import ensure_path_exists, synth_midi
from omnizart.constants.midi import SOUNDFONT_PATH
from omnizart.cli.music import music
from omnizart.cli.drum import drum
from omnizart.cli.chord import chord
from omnizart.cli.transcribe import transcribe


SUB_COMMAND_GROUP = [
    {"Transcription": ["music", "chord", "drum", "transcribe"]},
    {"Utilities": ["download-checkpoints", "download-dataset", "synth"]}
]


class GroupSubCommandHelpMsg(click.Group):
    """Group different types of sub-commands when showing help message."""
    def format_commands(self, ctx, formatter):
        all_commands = self.list_commands(ctx)
        limit = formatter.width - 6 - max(len(cmd) for cmd in all_commands)

        for group in SUB_COMMAND_GROUP:
            grp_name = list(group.keys())[0]
            subcommands = list(group.values())[0]
            rows = []
            for subcommand in subcommands:
                cmd = self.get_command(ctx, subcommand)
                help_msg = cmd.get_short_help_str(limit)
                rows.append((subcommand, help_msg))
                all_commands.remove(subcommand)

            with formatter.section(grp_name):
                formatter.write_dl(rows)

        other_cmd = []
        for subcommand in all_commands:
            cmd = self.get_command(ctx, subcommand)
            if cmd is None:
                continue
            if cmd.hidden:
                continue

            help_msg = cmd.get_short_help_str(limit)
            other_cmd.append((subcommand, help_msg))

        if len(other_cmd) > 0:
            with formatter.section("Others"):
                formatter.write_dl(other_cmd)


@click.group(cls=GroupSubCommandHelpMsg)
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
@click.option("--output-path", help="Explicitly specify the path to the omnizart project for storing checkpoints.")
def download_checkpoints(output_path):
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
            "file_length": 33813824
        },
        "music_piano": {
            "fid": "1x9_qjXSiM4GAxpvKfdYJK5S3SLdlCl2I",
            "save_as": "checkpoints/music/music_piano/weights.h5",
            "file_length": 50738464
        },
        "music_note_stream": {
            "fid": "18IqdrR3IhFP52H6w1HgTlPQUJXKHXm9u",
            "save_as": "checkpoints/music/music_note_stream/weights.h5",
            "file_length": 33816384
        }
    }

    if output_path is not None:
        abs_path = os.path.abspath(output_path)
        output_path = os.path.join(abs_path, "omnizart")
    else:
        output_path = MODULE_PATH

    for checkpoint, info in CHECKPOINTS.items():
        print(f"Downloading checkpoints: {checkpoint}")
        save_name = os.path.basename(info["save_as"])
        save_path = os.path.dirname(info["save_as"])
        save_path = os.path.join(output_path, save_path)
        download_large_file_from_google_drive(
            info["fid"], file_length=info["file_length"], save_path=save_path, save_name=save_name
        )


@click.command()
@click.argument("input_midi", type=click.Path(exists=True))
@click.option(
    "-o", "--output-path", help="Output path of the synthesized midi.", type=click.Path(writable=True), default="./"
)
@click.option("--sf2-path", help="Path to your own soundfont file.", type=click.Path(exists=True))
def synth(input_midi, output_path, sf2_path):
    """Synthesize the MIDI into wav file.

    If --sf2-path is not specified, will use the default soundfont file same as used by MuseScore."
    """
    f_name, _ = os.path.splitext(os.path.basename(input_midi))
    out_name = f"{f_name}_synth.wav"
    if os.path.isdir(output_path):
        # Specifies only directory without file name.
        # Use the default file name.
        ensure_path_exists(output_path)
        output_file = os.path.join(output_path, out_name)
    else:
        # Already specified the output file name.
        f_dir = os.path.dirname(os.path.abspath(output_path))
        ensure_path_exists(f_dir)
        output_file = output_path
    click.echo(f"Output file as: {output_file}")

    if sf2_path is None:
        if not os.path.exists(SOUNDFONT_PATH):
            # Download the default soundfont file.
            click.echo("Downloading default sondfont file...")
            download_large_file_from_google_drive(
                url="16RM-dWKcNtjpBoo7DFSONpplPEg5ruvO",
                file_length=31277462,
                save_path=os.path.dirname(SOUNDFONT_PATH),
                save_name=os.path.basename(SOUNDFONT_PATH)
            )
        sf2_path = SOUNDFONT_PATH

    click.echo("Synthesizing MIDI...")
    synth_midi(input_midi, output_path=output_file, sf2_path=sf2_path)
    click.echo("Synthesize finished")


entry.add_command(music)
entry.add_command(drum)
entry.add_command(chord)
entry.add_command(transcribe)
entry.add_command(download_dataset)
entry.add_command(download_checkpoints)
entry.add_command(synth)
