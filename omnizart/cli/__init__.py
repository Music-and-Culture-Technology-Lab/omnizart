"""Command line interface of `omnizart`
"""

from omnizart.music import app as music_app
from omnizart.drum import app as drum_app


apps = {
    "music": music_app,
    "drum": drum_app
}
