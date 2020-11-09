# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['omnizart',
 'omnizart.chord',
 'omnizart.cli',
 'omnizart.cli.chord',
 'omnizart.cli.drum',
 'omnizart.cli.music',
 'omnizart.constants',
 'omnizart.constants.schema',
 'omnizart.drum',
 'omnizart.feature',
 'omnizart.models',
 'omnizart.music']

package_data = \
{'': ['*'],
 'omnizart': ['checkpoints/chord/chord_v1/checkpoint',
              'checkpoints/chord/chord_v1/checkpoint',
              'checkpoints/chord/chord_v1/checkpoint',
              'checkpoints/chord/chord_v1/configurations.yaml',
              'checkpoints/chord/chord_v1/configurations.yaml',
              'checkpoints/chord/chord_v1/configurations.yaml',
              'checkpoints/chord/chord_v1/weights.index',
              'checkpoints/chord/chord_v1/weights.index',
              'checkpoints/chord/chord_v1/weights.index',
              'checkpoints/drum/drum_keras/arch.yaml',
              'checkpoints/drum/drum_keras/arch.yaml',
              'checkpoints/drum/drum_keras/configurations.yaml',
              'checkpoints/drum/drum_keras/configurations.yaml',
              'checkpoints/music/music_note_stream/arch.yaml',
              'checkpoints/music/music_note_stream/arch.yaml',
              'checkpoints/music/music_note_stream/configurations.yaml',
              'checkpoints/music/music_note_stream/configurations.yaml',
              'checkpoints/music/music_piano/arch.yaml',
              'checkpoints/music/music_piano/arch.yaml',
              'checkpoints/music/music_piano/arch.yaml',
              'checkpoints/music/music_piano/configuration.json',
              'checkpoints/music/music_piano/configuration.json',
              'checkpoints/music/music_piano/configuration.json',
              'checkpoints/music/music_piano/configurations.yaml',
              'checkpoints/music/music_piano/configurations.yaml',
              'checkpoints/music/music_piano/configurations.yaml',
              'checkpoints/music/music_pop/arch.yaml',
              'checkpoints/music/music_pop/arch.yaml',
              'checkpoints/music/music_pop/configurations.yaml',
              'checkpoints/music/music_pop/configurations.yaml',
              'defaults/*',
              'resource/vamp/*']}

install_requires = \
['click>=7.1.2,<8.0.0',
 'jsonschema>=3.2.0,<4.0.0',
 'librosa>=0.8.0,<0.9.0',
 'madmom>=0.16.1,<0.17.0',
 'numba==0.48',
 'opencv-python>=4.4.0,<5.0.0',
 'pretty_midi>=0.2.9,<0.3.0',
 'pyfluidsynth>=1.2.5,<2.0.0',
 'pyyaml>=5.3.1,<6.0.0',
 'tensorflow>=2.3.0,<3.0.0',
 'tqdm>=4.49.0,<5.0.0',
 'urllib3>=1.25.11,<2.0.0',
 'vamp>=1.1.0,<2.0.0']

entry_points = \
{'console_scripts': ['omnizart = omnizart.cli.cli:entry']}

setup_kwargs = {
    'name': 'omnizart',
    'version': '0.1.0b1',
    'description': 'Omniscient Mozart, being able to transcribe everything in the music.',
    'long_description': "# omnizart\n\n![build](https://github.com/Music-and-Culture-Technology-Lab/omnizart/workflows/general-check/badge.svg)\n[![docs](https://github.com/Music-and-Culture-Technology-Lab/omnizart/workflows/docs/badge.svg?branch=build_doc)](https://music-and-culture-technology-lab.github.io/omnizart-doc/)\n[![PyPI version](https://badge.fury.io/py/omnizart.svg)](https://badge.fury.io/py/omnizart)\n![PyPI - License](https://img.shields.io/pypi/l/omnizart)\n[![PyPI - Downloads](https://img.shields.io/pypi/dm/omnizart)](https://pypistats.org/packages/omnizart)\n[![Docker Pulls](https://img.shields.io/docker/pulls/mctlab/omnizart)](https://hub.docker.com/r/mctlab/omnizart)\n\nOmniscient Mozart, being able to transcribe everything in the music, including vocal, drum, chord, beat, instruments, and more.\nCombines all the hard works developed by everyone in MCTLab into a single command line tool, and plan to distribute as a python package in the future.\n\nA quick-start example is as following:\n``` bash\n# Install omnizart\npip install omnizart\n\n# Download the checkpoints after installation\nomnizart download-checkpoints\n\n# Now it's ready for the transcription~\nomnizart drum transcribe <path/to/audio.wav>\nomnizart chord transcribe <path/to/audio.wav>\nomnizart music transcribe <path/to/audio.wav>\n```\n\nOr use the docker image:\n```\ndocker pull mctlab/omnizart:latest\ndocker run -it mctlab/omnizart:latest bash\n```\n\nComprehensive usage and API references can be found in the [official documentation site](https://music-and-culture-technology-lab.github.io/omnizart-doc/).\n\n# About\n[Music and Culture Technology Lab (MCTLab)](https://sites.google.com/view/mctl/home) aims to develop technology for music and relevant applications by leveraging cutting-edge AI techiniques.\n\n# Plan to support\n| Commands | transcribe         | train              | evaluate | Description                       |\n|----------|--------------------|--------------------|----------|-----------------------------------|\n| music    | :heavy_check_mark: | :heavy_check_mark: |          | Transcribes notes of instruments. |\n| drum     | :heavy_check_mark: | :interrobang:      |          | Transcribes drum tracks.          |\n| vocal    |                    |                    |          | Transcribes pitch of vocal.       |\n| chord    | :heavy_check_mark: | :heavy_check_mark: |          | Transcribes chord progression.    |\n| beat     |                    |                    |          | Transcribes beat position.        |\n\n**NOTES** Though the implementation of training the drum model is 90% complete, but there still exists some\ninvisible bugs that cause the training fails to converge compared to the author's original implementation.\n\nExample usage\n<pre>\nomnizart music transcribe <i>path/to/audio</i>\nomnizart chord transcribe <i>path/to/audio</i>\nomnizart drum transcribe <i>path/to/audio</i>\n</pre>\n\nFor training a new model, download the dataset first and follow steps described below.\n<pre>\n# The following command will default saving the extracted feature under the same folder,\n# called <b>train_feature</b> and <b>test_feature</b>\nomnizart music generate-featuer -d <i>path/to/dataset</i>\n\n# Train a new model\nomnizart music train-model -d <i>path/to/dataset</i>/train_feature --model-name My-Model\n</pre>\n\n\n# Development\nDescribes the neccessary background of how to develop this project.\n\n## Download and install\n``` bash\ngit clone https://github.com/Music-and-Culture-Technology-Lab/omnizart.git\n\n# Install dependenies. For more different installation approaches, please refer to the official documentation page.\n# The following command will download the checkpoints automatically.\ncd omnizart\nmake install\n\n# For developers, you have to install Dev dependencies as well, since they will not be installed by default.\npoetry install\n```\n\n## Package management\nUses [poetry](https://python-poetry.org/) for package management, instead of writing `requirements.txt` and `setup.py` manually.\nWe still provide the above two files for convenience. You can also generate them by executing ``make export``.\n\n## Documentation\nAutomatically generate documents from inline docstrings of module, class, and function. \n[Hosted document page](http://140.109.21.96:8000/build/html/index.html)\n\nDocumentation style: Follows `numpy` document flavor. Learn more from [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).\n\nDocument builder: [sphinx](https://www.sphinx-doc.org/en/master/)\n\nTo generate documents, `cd docs/` and execute `make html`. To see the rendered results, run `make serve` and view from the browser.\nAll documents and docstrings use **reStructured Text** format. More informations about this format can be found from \n[Sphinx's Document](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).\n\n## Linters\nUses flake8 and pylint for coding style check.\n\nTo check with linters, execute `make check`.\n\nYou don't have to achieve a perfect score on pylint check, just pass 9.5 points still counted as a successful check.\n\n### Caution!\nThere is convenient make command for formating the code, but it should be used very carefully.\nNot only it could format the code for you, but also could mess up the code, and eventually you should still need\nto check the format manually after refacorting the code with tools. \n\nTo format the code with black and yapf, enter `make format`.\n\n## Unittest\nUses `pytest` for unittesting. Under construction...\n\n## CI/CD\nUses github actions for automatic linting, unittesting, document building, and package release.\nCurrently supports two workflows:\n* General check\n* Documentation page publishing\n* Publish PyPI package and docker image\n\n### General Check\nEverytime you push to the master branch, file a pull request, and merge into master branch, will trigger\nthis action. This will do checks like code format, and unittests by leveraging the above mentioned\ntools. If the check fails, you will not be able to merge the feature branch into master branch.\n\n### Documentation Page Publishing\nWe use [github page](https://pages.github.com/) to host our documentation, and is separated as an [independent\nrepository](https://github.com/Music-and-Culture-Technology-Lab/omnizart-doc). \n\n**Please do not directly modify the content of the omnizart-doc repository!!**\n\nThe only permitted way to update the documentation page is by updating the `build_doc` branch, and\nlet the workflow do the rest of things.\n\nSteps to update the documentation page:\n* Clone **this** repo\n* Create a new branch. **DO NOT UPDATE THE `build_doc` BRANCH DIRECTLY!!**\n* File a pull request\n* Merge into master (by admin)\n* Merge into `build_doc` branch (by admin)\n* Push to this repo (by admin)\n\n### Publish PyPI Package and Docker Image\nPublish the python package to PyPI and also the docker image to dockerhub when push tags to the repository.\nThe publish process will be automatically done by the github actions. There are several steps in the process:\n\n1. Pack and publish the python package.\n2. Build the docker image and publish to Docker Hub.\n3. Create release -> this will also trigger the automation of documentation publishment.\n\n\n## Docker\nWe provide both the Dockerfile for local image build and also the pre-build image on Docker Hub.\n\nTo build the image, run the following:\n```\ndocker build -t omnizart:my-image .\n```\n\nTo use the pre-build image, follow below steps:\n```\n# Download from Docker Hub\ndocker pull mctlab/omnizart\n\n# Execute the image\ndocker run -it mctlab/omnizart:latest\n\n### For those who want to leverage the power of GPU for acceleration, make sure\n### you have installed docker>=19.03 and the 'nvidia-container-toolkit' package.\n# Execute the docker with GPU support\ndocker run --gpus all -it mctlab/omnizart:latest\n```\n\n\n## Command Test\nTo actually install and test the `omnizart` command, execute `make install`. This will automatically create a virtual environment and install everything needed inside it. After installation, just follow the instruction showing on the screen to activate the environment, then type `omnizart --help` to check if it works. After testing the command, type `deactivate` to leave the virtual environment. \n",
    'author': 'BreezeWhite',
    'author_email': 'freedombluewater@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://sites.google.com/view/mctl/home',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.6,<4',
}


setup(**setup_kwargs)

# This setup.py was autogenerated using poetry.
