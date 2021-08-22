#!/bin/bash

set -e

# Script for installing the 'omnizart' command and the required packages.
#
# The script will first create a virtual environment, separating from the system's environment.
# There are two approaches for installing the 'omnizart' pacakage: poetry or pip. 
# The former relies on the third-party package 'poetry', and the latter is a built-in package.
# This script uses 'python -m venv .venv' to create the virutal environment for both
# installation approaches.
#
# After creating virtual env with venv, the script automatically installs the
# required packages and the 'omnizart' command in virtual env. As the installation
# finished, just activate the environment and enjoy~.


USE_VENV=false
if [ ! -z "$1" ] && [ "$1" = "venv"  ]; then
    USE_VENV=true
fi

INSTALL_APPROACH="${DEFAULT_INSTALL_APPROACH:=poetry}"
if [ "$USE_VENV" = "true" ]; then echo "Using $INSTALL_APPROACH to create virtual environment"; fi


upgrade_pkg() {
    python3 -m pip install --upgrade pip

    # Some packages have some problem installing with poetry.
    # Thus manually install them here.
    pip install --upgrade setuptools
    pip install wheel
}


activate_venv_with_poetry() {
    # Create virtual environment.
    poetry shell

    # Hacky way to activate the virtualenv due to some 
    # problem that exists in poetry.
    source $(dirname $(poetry run which python))/activate
}

activate_venv_with_venv() {
    python3 -m venv .venv
    source .venv/bin/activate
}

install_with_poetry() {
    if [ "$USE_VENV" = "false" ]; then
        poetry config virtualenvs.create false
        poetry config virtualenvs.in-project false
    fi
    poetry install --no-dev
}

install_with_pip() {
    # Install some tricky packages that cannot be resolved by setup.py
    # and requirements.txt.
    pip install Cython numpy
    pip install madmom --use-feature=2020-resolver

    pip install -r requirements.txt
    python3 setup.py install 
}


check_if_venv_activated() {
    # Check if virtual environment was successfully activated.
    if [ -z $VIRTUAL_ENV ]; then
        >&2 echo "Fail to activate virtualenv..."
        exit 1
    else
        echo "Successfully activate virtualenv";
    fi
}


# ------------ Start Installation ------------ #
# Need to upgrade pip first, or latter installation may fail.
upgrade_pkg

if [ "$INSTALL_APPROACH" = "poetry"  ]; then
    # Check if poetry is installed
    if ! hash poetry 2>/dev/null; then
        echo "Installing poetry..."
        pip install poetry
    fi

    if [ "$USE_VENV" = "true" ]; then
        activate_venv_with_venv
        check_if_venv_activated
        upgrade_pkg
    fi
    install_with_poetry
elif [ "$INSTALL_APPROACH" = "pip" ]; then
    if [ "$USE_VENV" = "venv" ]; then
        activate_venv_with_venv
        check_if_venv_activated
        upgrade_pkg
    fi
    install_with_pip
else
    >$2 echo "Unknown virtualenv method: $VENV_APPROACH"
    exit 1
fi

omnizart download-checkpoints

if [ "$USE_VENV" = "true" ]; then
    echo -e "\nTo activate the environment, run the following command:\n source .venv/bin/activate"
fi
