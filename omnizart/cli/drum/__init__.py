import click

from omnizart.cli.drum.transcribe import transcribe


@click.group()
def drum():
    """Transcribe drum percussions."""


drum.add_command(transcribe)
