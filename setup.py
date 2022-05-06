from setuptools import setup

packages = \
['omnizart',
 'omnizart.beat',
 'omnizart.chord',
 'omnizart.cli',
 'omnizart.cli.beat',
 'omnizart.cli.chord',
 'omnizart.cli.drum',
 'omnizart.cli.music',
 'omnizart.cli.patch_cnn',
 'omnizart.cli.vocal',
 'omnizart.cli.vocal_contour',
 'omnizart.constants',
 'omnizart.constants.schema',
 'omnizart.drum',
 'omnizart.feature',
 'omnizart.models',
 'omnizart.music',
 'omnizart.patch_cnn',
 'omnizart.vocal',
 'omnizart.vocal_contour']

package_data = \
{'': ['*'],
 'omnizart': ['checkpoints/beat/beat_blstm/*',
              'checkpoints/beat/beat_blstm/variables/variables.index',
              'checkpoints/chord/chord_v1/configurations.yaml',
              'checkpoints/chord/chord_v1/configurations.yaml',
              'checkpoints/chord/chord_v1/saved_model.pb',
              'checkpoints/chord/chord_v1/saved_model.pb',
              'checkpoints/chord/chord_v1/variables/variables.index',
              'checkpoints/drum/drum_keras/*',
              'checkpoints/drum/drum_keras/variables/variables.index',
              'checkpoints/music/music_note_stream/*',
              'checkpoints/music/music_note_stream/variables/variables.index',
              'checkpoints/music/music_piano-v2/*',
              'checkpoints/music/music_piano-v2/variables/variables.index',
              'checkpoints/music/music_piano/*',
              'checkpoints/music/music_piano/variables/variables.index',
              'checkpoints/music/music_pop/*',
              'checkpoints/music/music_pop/variables/variables.index',
              'checkpoints/patch_cnn/patch_cnn_melody/*',
              'checkpoints/patch_cnn/patch_cnn_melody/variables/variables.index',
              'checkpoints/vocal/vocal_contour/*',
              'checkpoints/vocal/vocal_contour/variables/variables.index',
              'checkpoints/vocal/vocal_semi/*',
              'checkpoints/vocal/vocal_semi/variables/variables.index',
              'defaults/*',
              'resource/vamp/*']}

install_requires = \
['click>=7.1.2,<8.0.0',
 'jsonschema>=3.2.0,<4.0.0',
 'librosa>=0.8.0,<0.9.0',
 'madmom>=0.16.1,<0.17.0',
 'mir_eval>=0.6,<0.7',
 'pillow>=8.3.2,<9.0.0',
 'pretty_midi>=0.2.9,<0.3.0',
 'pyfluidsynth>=1.2.5,<2.0.0',
 'pyyaml>=5.3.1,<6.0.0',
 'spleeter>=2.3.0,<3.0.0',
 'tensorflow==2.5.0',
 'tqdm>=4.49.0,<5.0.0',
 'urllib3==1.26.4',
 'vamp>=1.1.0,<2.0.0']

entry_points = \
{'console_scripts': ['omnizart = omnizart.cli.cli:entry']}

setup_kwargs = {
    'name': 'omnizart',
    'version': '0.5.0',
    'description': 'Omniscient Mozart, being able to transcribe everything in the music.',
    'long_description': '# OMNIZART\n\n[![build](https://github.com/Music-and-Culture-Technology-Lab/omnizart/workflows/general-check/badge.svg)](https://github.com/Music-and-Culture-Technology-Lab/omnizart/actions?query=workflow%3Ageneral-check)\n[![docs](https://github.com/Music-and-Culture-Technology-Lab/omnizart/workflows/docs/badge.svg?branch=build_doc)](https://music-and-culture-technology-lab.github.io/omnizart-doc/)\n[![PyPI version](https://badge.fury.io/py/omnizart.svg)](https://badge.fury.io/py/omnizart)\n![PyPI - License](https://img.shields.io/pypi/l/omnizart)\n[![Downloads](https://static.pepy.tech/personalized-badge/omnizart?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads/month)](https://pepy.tech/project/omnizart)\n[![Docker Pulls](https://img.shields.io/docker/pulls/mctlab/omnizart)](https://hub.docker.com/r/mctlab/omnizart)\n\nOmnizart is a Python library that aims for democratizing automatic music transcription.\nGiven polyphonic music, it is able to transcribe pitched instruments, vocal melody, chords, drum events, and beat.\nThis is powered by the research outcomes from [Music and Culture Technology (MCT) Lab](https://sites.google.com/view/mctl/home).\n\n### Transcribe your favorite songs now in Colab! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/OmnizartColab)\n\n# Quick start\n\nVisit the [complete document](https://music-and-culture-technology-lab.github.io/omnizart-doc/) for detailed guidance.\n\n## Pip\n``` bash\n# Install omnizart\npip install omnizart\n\n# Download the checkpoints\nomnizart download-checkpoints\n\n# Transcribe your songs\nomnizart drum transcribe <path/to/audio.wav>\nomnizart chord transcribe <path/to/audio.wav>\nomnizart music transcribe <path/to/audio.wav>\n```\n\n## Docker\n```\ndocker pull mctlab/omnizart:latest\ndocker run -it mctlab/omnizart:latest bash\n```\n\n# Supported applications\n| Application      | Transcription      | Training           | Evaluation | Description                                      |\n|------------------|--------------------|--------------------|------------|--------------------------------------------------|\n| music            | :heavy_check_mark: | :heavy_check_mark: |            | Transcribe musical notes of pitched instruments. |\n| drum             | :heavy_check_mark: | :interrobang:      |            | Transcribe events of percussive instruments.     |\n| vocal            | :heavy_check_mark: | :heavy_check_mark: |            | Transcribe note-level vocal melody.              |\n| vocal-contour    | :heavy_check_mark: | :heavy_check_mark: |            | Transcribe frame-level vocal melody (F0).        |\n| chord            | :heavy_check_mark: | :heavy_check_mark: |            | Transcribe chord progressions.                   |\n| beat             | :heavy_check_mark: | :heavy_check_mark: |            | Transcribe beat position.                        |\n\n**NOTES**\nThe current implementation for the drum model has unknown bugs, preventing loss convergence when training from scratch.\nFortunately, you can still enjoy drum transcription with the provided checkpoints.\n\n## Compatibility Issue\nCurrently, Omnizart is **incompatible for ARM-based MacOS** system due to the underlying dependencies.\nMore details can be found in the [issue #38](https://github.com/Music-and-Culture-Technology-Lab/omnizart/issues/38).\n',
    'author': 'BreezeWhite',
    'author_email': 'miyashita2010@tuta.io',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://sites.google.com/view/mctl/home',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.6.1,<3.9',
}


setup(**setup_kwargs)

# This setup.py was autogenerated using poetry.
