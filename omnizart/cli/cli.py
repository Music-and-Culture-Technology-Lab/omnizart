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


LAZY_COMMANDS = {
    "music": ("omnizart.cli.music", "music", "Transcribe instruments and corresponding pitch in the audio."),
    "drum": ("omnizart.cli.drum", "drum", "Transcribe drum percussions."),
    "chord": ("omnizart.cli.chord", "chord", "Transcribe chord progression"),
    "vocal": ("omnizart.cli.vocal", "vocal", "Transcribe vocal notes in the audio."),
    "vocal-contour": ("omnizart.cli.vocal_contour", "vocal_contour", "Transcribe vocal melody (frame-based) in the audio."),
    "beat": ("omnizart.cli.beat", "beat", "Beat tracking on symbolic domain."),
    "patch-cnn": ("omnizart.cli.patch_cnn", "patch_cnn", "Trancribes vocal melody (frame-based) in the audio."),
    "transcribe": ("omnizart.cli.transcribe", "transcribe", "Transcribe a single audio."),
}


SUB_COMMAND_GROUP = [
    {
        "Transcription": [
            "music", "chord", "drum", "vocal", "vocal-contour", "beat", "patch-cnn", "transcribe"
        ]
    },
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
                if subcommand in LAZY_COMMANDS:
                    help_msg = LAZY_COMMANDS[subcommand][2]
                else:
                    cmd = self.get_command(ctx, subcommand)
                    help_msg = cmd.get_short_help_str(limit) if cmd else ""
                rows.append((subcommand, help_msg))
                if subcommand in all_commands:
                    all_commands.remove(subcommand)

            with formatter.section(grp_name):
                formatter.write_dl(rows)

        other_cmd = []
        for subcommand in all_commands:
            if subcommand in LAZY_COMMANDS:
                help_msg = LAZY_COMMANDS[subcommand][2]
            else:
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

    def get_command(self, ctx, name):
        cmd = self.commands.get(name)
        if cmd is not None:
            return cmd

        if name in LAZY_COMMANDS:
            import importlib
            mod_path, var_name, _ = LAZY_COMMANDS[name]
            mod = importlib.import_module(mod_path)
            cmd = getattr(mod, var_name)
            self.add_command(cmd, name)
            return cmd

        return None

    def list_commands(self, ctx):
        return sorted(list(set(LAZY_COMMANDS.keys()).union(self.commands.keys())))


@click.group(cls=GroupSubCommandHelpMsg)
def entry():
    pass


@click.command()
@click.argument(
    "dataset",
    type=click.Choice(["Maestro", "MusicNet", "McGill", "BPS-FH", "Ext-Su", "MIR1K", "CMedia"], case_sensitive=False)
)
@click.option(
    "-o", "--output", default="./", help="Path for saving the downloaded dataset.", type=click.Path(writable=True)
)
def download_dataset(dataset, output):
    """A quick command for downloading datasets."""
    import omnizart.constants.datasets as dset
    struct = {
        "maestro": dset.MaestroStructure,
        "musicnet": dset.MusicNetStructure,
        "mcgill": dset.McGillBillBoard,
        "bps-fh": dset.BeethovenSonatasStructure,
        "ext-su": dset.ExtSuStructure,
        "mir1k": dset.MIR1KStructure,
        "cmedia": dset.CMediaStructure
    }[dataset.lower()]
    click.echo(f"Downloading {dataset} dataset and save to {output}")
    struct.download(save_path=output)


@click.command()
@click.option("--output-path", help="Explicitly specify the path to the omnizart project for storing checkpoints.")
def download_checkpoints(output_path):
    """Download the archived checkpoints of different models."""
    from omnizart import MODULE_PATH
    from omnizart.remote import download_large_file_from_google_drive

    release_url = "https://github.com/Music-and-Culture-Technology-Lab/omnizart/releases/download/checkpoints-20211001"
    CHECKPOINTS = {
        "chord_v1": {
            "fid": f"{release_url}/chord_v1@variables.data-00000-of-00001",
            "save_as": "checkpoints/chord/chord_v1/variables/variables.data-00000-of-00001"
        },
        "drum_keras": {
            "fid": f"{release_url}/drum_keras@variables.data-00000-of-00001",
            "save_as": "checkpoints/drum/drum_keras/variables/variables.data-00000-of-00001",
        },
        "music_pop": {
            "fid": f"{release_url}/music_pop@variables.data-00000-of-00001",
            "save_as": "checkpoints/music/music_pop/variables/variables.data-00000-of-00001",
        },
        "music_piano": {
            "fid": f"{release_url}/music_piano@variables.data-00000-of-00001",
            "save_as": "checkpoints/music/music_piano/variables/variables.data-00000-of-00001",
        },
        "music_piano-v2": {
            "fid": f"{release_url}/music_piano-v2@variables.data-00000-of-00001",
            "save_as": "checkpoints/music/music_piano-v2/variables/variables.data-00000-of-00001",
        },
        "music_note_stream": {
            "fid": f"{release_url}/music_note_stream@variables.data-00000-of-00001",
            "save_as": "checkpoints/music/music_note_stream/variables/variables.data-00000-of-00001",
        },
        "vocal_semi": {
            "fid": f"{release_url}/vocal_semi@variables.data-00000-of-00001",
            "save_as": "checkpoints/vocal/vocal_semi/variables/variables.data-00000-of-00001",
        },
        "vocal_contour": {
            "fid": f"{release_url}/contour@variables.data-00000-of-00001",
            "save_as": "checkpoints/vocal/vocal_contour/variables/variables.data-00000-of-00001",
        },
        "beat": {
            "fid": f"{release_url}/beat_blstm@variables.data-00000-of-00001",
            "save_as": "checkpoints/beat/beat_blstm/variables/variables.data-00000-of-00001",
        },
        "patch_cnn_melody": {
            "fid": f"{release_url}/patch_cnn_melody@variables.data-00000-of-00001",
            "save_as": "checkpoints/patch_cnn/patch_cnn_melody/variables/variables.data-00000-of-00001",
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
        unzip = info.get("unzip", False)
        download_large_file_from_google_drive(
            info["fid"],
            file_length=info.get("file_length", None),
            save_path=save_path,
            save_name=save_name,
            unzip=unzip
        )
        if unzip:
            os.remove(os.path.join(save_path, save_name))


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
    from omnizart.constants.midi import SOUNDFONT_PATH
    from omnizart.remote import download_large_file_from_google_drive
    from omnizart.utils import ensure_path_exists, synth_midi

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
                url="https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2",
                file_length=215614036,
                save_path=os.path.dirname(SOUNDFONT_PATH),
                save_name=os.path.basename(SOUNDFONT_PATH)
            )
        sf2_path = SOUNDFONT_PATH

    click.echo("Synthesizing MIDI...")
    synth_midi(input_midi, output_path=output_file, sf2_path=sf2_path)
    click.echo("Synthesize finished")


entry.add_command(download_dataset)
entry.add_command(download_checkpoints)
entry.add_command(synth)
