Quick Start
===========

Colab
#####

Play with the `Colab notebook <https://bit.ly/omnizart-colab>`_  to transcribe your favorite song without hassles.
You can also follow the installation below to enjoy Omnizart locally.

Installation
############

Using pip
*********

Omnizart is under development and will be updated regularly on PyPI.
Use ``pip`` to install the latest stable version.

.. code-block:: bash

    # Install the prerequisites manually since there are some dependencies can't be
    # resolved automatically.
    pip install numpy Cython

    # Additional system packages are required to fully use Omnizart.
    sudo apt-get install libsndfile-dev fluidsynth ffmpeg

    # Install Omnizart
    pip install omnizart

    # Then download the checkpoints
    omnizart download-checkpoints


Development installation
************************

For the development installation, clone the git repo and the installation
creates a virtual environment under the directory *omnizart/* by default.

.. code-block:: bash

    # Clone the omnizart repository from GitHub
    git clone https://github.com/Music-and-Culture-Technology-Lab/omnizart.git

    # Install dependencies, with checkpoints automatically downloaded
    cd omnizart
    make install

    # Install Dev dependencies, since they will not be installed by default
    poetry install


CLI
###

Below is an example usage of pitched instrument transcription with the command-line interface,
first transcribing a piece of music and then synthesizing the results.
For more details and other types of transcription, refer to :doc:`tutorial`.

Transcription
*************

The example transcribes a piece of music, being monophonic or polyphonic,
to a MIDI file of the transcribed pitched notes and a CSV file with more information.

.. code-block:: bash

    omnizart music transcribe <path/to/example.wav>


Sonification
************

Omnizart renders the transcribed MIDI file with default soundfonts,
synthesizing an audio in WAV by the command below.
For the first-time execution, it is expected to take a bit for downloading the free-licensed soundfonts.

.. code-block:: bash

    omnizart synth example.mid


Auto Completion
***************

To enable auto completion, type the following according to your environment type.

.. code-block:: bash

    # For bash
    _OMNIZART_COMPLETE=source_bash omnizart > omnizart-complete.sh

    # For zsh
    _OMNIZART_COMPLETE=source_zsh omnizart > omnizart-complete.sh

    # Source the generated script to enable
    source omnizart-complete.sh
