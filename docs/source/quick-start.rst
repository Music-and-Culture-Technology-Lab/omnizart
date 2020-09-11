Quick Start
===========

Installation
############

Using pip
*********

.. code-block:: bash

    pip install omnizart


From source
***********

The most guaranteed way for installing the pacakge from source.
This will create a virtual environment under the same folder,
and thus not installed system-wide. For the system-wide installation
approach, please refer to the following section.

.. code-block:: bash

    git clone https://github.com/Music-and-Culture-Technology-Lab/omnizart.git
    cd omnizart
    make install



The second way is by using the provided shell script to install, which
is almost identical to execute ``make install`` command, except you
can specify more settings for the installtoin including to install
system-wide this time.

.. code-block:: bash

    ### Under the `omnizart` folder
    # System-wide installation
    ./scripts/install.sh

    # Install with virtual environment
    ./scripts/install.sh venv

    # Use 'poetry' for installation
    export DEFAULT_VENV_APPROACH=poetry
    ./scripts/install.sh

    # Use built-in 'venv' libarary for installation
    export DEFAULT_VENV_APPROACH=venv
    ./scripts/install.sh



A more manual way for installing this package is by using ``poetry``, which we use this
tool for the dependency management.

.. code-block:: bash

    # Under the `omnizart` folder
    pip install --upgrade pip
    pip install setuptools==50.0.3
    pip install poetry
    poetry install


The last and the most unstable, conventional way for installation
is by using the ``setup.py`` and ``requirements.txt`` files.

.. code-block:: bash

    # Under the `omnizart` folder
    pip install --upgrade pip
    pip install setuptools==50.0.3

    # May encounter problems when installing the dependency 'madmom'
    python setup.py install


Transcribe a pop song
#####################

Transcribes a song into a MIDI file and a CSV file with more complete
and representative information.

.. code-block:: bash

    omnizart transcribe all <path/to/audio.wav>

