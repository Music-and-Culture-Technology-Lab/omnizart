import click

from omnizart.cli.drum.transcribe import transcribe
from omnizart.cli.drum.generate_feature import generate_feature


@click.group()
def drum():
    """Transcribe drum percussions."""


drum.add_command(transcribe)
drum.add_command(generate_feature)
