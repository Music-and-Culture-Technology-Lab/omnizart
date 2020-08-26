#!/bin/bash

# Script for installing the 'omnizart' command and the required packages.
#
# The script will first create the virtual environment, separating the system's environment.
# There are two approaches of creating the virtual env: poetry and venv. 
# The former is a third-party library, and the latter is the built-in library.
# The script use venv as the default tool. You could setup environment variable
# 'DEFAULT_VENV_APPROACH' to 'poetry' to use poetry for both create virtual env and
# installation.
#
# After creating the virtual env with venv, the script automatically installs the
# required packages and the 'omnizart' command in virtual env. As the installation
# finished, just activate the environment and enjoy~.


VENV_APPROACH="${DEFAULT_VENV_APPROACH:=poetry}"
echo "Using $VENV_APPROACH to create virtual environment"


activate_venv_with_poetry() {
    if ! hash poetry 2>/dev/null; then
        # Poetry haven't been installed, install it first.
        echo "Install poetry..."
        python3 -m pip install poetry
    fi

    # Create virtual environment.
    poetry shell > /dev/null
    
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
    python3 -m pip install --upgrade pip

    # This package has some problem installing with pip.
    # Thus use easy_install here instead.
    python3 -m pip install --upgrade setuptools
    python3 -m easy_install llvmlite
}

install_with_poetry() {
    pre_install
    poetry install
}

install_with_pip() {
    pre_install
    python3 -m pip install -r requirements.txt
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
    activate_venv_with_poetry
    check_if_venv_activated
    install_with_poetry

    echo -e "\nTo activate the environment, run the following command:"
    echo "source \$(dirname \$(poetry run which python))/activate"
elif [ "$VENV_APPROACH" = "venv" ]; then
    activate_venv_with_venv
    check_if_venv_activated
    install_with_pip

    echo -e "\nTo activate the environment, run the following command:"
    echo "source .venv/bin/activate"
else
    >$2 echo "Unknown virtualenv method: $VENV_APPROACH"
    exit 1
fi

