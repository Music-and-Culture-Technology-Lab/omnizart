#!/bin/bash

set -e

# Script for installing the 'omnizart' command and the required packages.
#
# The script will first create a virtual environment, separating from the system's environment.
# There are two approaches for creating the virtual env: poetry or venv. 
# The former relies on the third-party package 'poetry', and the latter is a built-in package.
# This script uses venv as the default tool. You could also setup an environment variable
# 'DEFAULT_VENV_APPROACH' to 'poetry' for using poetry to create virtual env and
# install packages.
#
# After creating virtual env with venv, the script automatically installs the
# required packages and the 'omnizart' command in virtual env. As the installation
# finished, just activate the environment and enjoy~.


USE_VENV=false
if [ ! -z "$1" ] && [ "$1" = "venv"  ]; then
    USE_VENV=true
fi

VENV_APPROACH="${DEFAULT_VENV_APPROACH:=poetry}"
if [ "$USE_VENV" = "true" ]; then echo "Using $VENV_APPROACH to create virtual environment"; fi



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

pre_install() {
    # Need to upgrade pip first, or latter installation may fail.
    pip install --upgrade pip

    # Some packages have some problem installing with poetry.
    # Thus manually install them here.
    pip install setuptools==50.0.3
}

install_with_poetry() {
    pre_install
    if ! hash poetry 2>/dev/null; then
        echo "Installing poetry..."
        pip install poetry
    fi
    
    if [ "$USE_VENV" = "false" ]; then
        poetry config virtualenvs.create false
    fi
    poetry install --no-dev
}

install_with_pip() {
    pre_install
    
    # Install some tricky packages that cannot be resolved by setup.py
    # and requirements.txt.
    pip install madmom --use-feature=2020-resolver

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


if [ "$VENV_APPROACH" = "poetry"  ]; then
    if [ "$USE_VENV" = "true" ]; then
        activate_venv_with_poetry
        check_if_venv_activated
    fi
    install_with_poetry

    echo -e "\nTo activate the environment, run the following command:"
    echo "source \$(dirname \$(poetry run which python))/activate"
elif [ "$VENV_APPROACH" = "venv" ]; then
    if [ "$USE_VENV" = "venv" ]; then
        activate_venv_with_venv
        check_if_venv_activated
    fi
    install_with_pip

    echo -e "\nTo activate the environment, run the following command:"
    echo "source .venv/bin/activate"
else
    >$2 echo "Unknown virtualenv method: $VENV_APPROACH"
    exit 1
fi

