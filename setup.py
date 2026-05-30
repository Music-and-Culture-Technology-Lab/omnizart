from setuptools import setup
import sys
import os
import subprocess
import shutil


def install_packages(packages, no_build_isolation=False):
    # Locate package installer (prefer uv, then pip)
    uv_path = shutil.which("uv")
    pip_path = os.path.join(os.path.dirname(sys.executable), "pip")
    if not os.path.exists(pip_path):
        pip_path = shutil.which("pip") or "pip"

    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("PYTHONNOUSERSITE", None)

    if uv_path:
        cmd = [uv_path, "pip", "install", "--python", sys.executable]
    else:
        cmd = [pip_path, "install"]

    if no_build_isolation:
        cmd.append("--no-build-isolation")

    cmd.extend(packages)
    try:
        subprocess.check_call(cmd, env=clean_env)
        return True
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"Installer failed to install {packages}: {e}\n")
        return False


# Pre-install Cython, numpy, madmom, vamp, and pyaudio to avoid PEP 517 build isolation issues
try:
    import Cython
    import numpy
    # import madmom
    # import vamp
except ImportError:
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("PYTHONNOUSERSITE", None)

    # Step 1: Pre-install build tools Cython and numpy
    if not install_packages(["Cython>=0.29.32", "numpy>=1.19.0", "setuptools<82"]):
        raise RuntimeError("ERROR: Failed to pre-install build tools Cython and numpy.")

try:
    import madmom
    import vamp
except ImportError:
    # Step 2: Install madmom and vamp with --no-build-isolation since they require Cython and numpy at build-time
    if not install_packages(["madmom>=0.16.1"], no_build_isolation=True):
        raise RuntimeError("ERROR: Failed to pre-install dependency 'madmom'.")
    if not install_packages(["vamp>=1.1.0"], no_build_isolation=True):
        raise RuntimeError("ERROR: Failed to pre-install dependency 'vamp'.")

# Step 3: Install pyaudio, with a conda fallback and detailed error message for missing system portaudio
pyaudio_installed = False
try:
    import pyaudio
    pyaudio_installed = True
except ImportError:
    pyaudio_installed = install_packages(["pyaudio"])
    if not pyaudio_installed:
        # Fallback to conda if available
        conda_path = os.environ.get("CONDA_EXE")
        if conda_path and os.path.exists(conda_path):
            try:
                env_prefix = os.path.dirname(os.path.dirname(sys.executable))
                clean_env = os.environ.copy()
                clean_env.pop("PYTHONPATH", None)
                clean_env.pop("PYTHONNOUSERSITE", None)
                subprocess.check_call([conda_path, "install", "-y", "-p", env_prefix, "-c", "conda-forge", "pyaudio"], env=clean_env)
                pyaudio_installed = True
            except Exception as e:
                sys.stderr.write(f"Conda fallback install for pyaudio failed: {e}\n")
        
        if not pyaudio_installed:
            raise RuntimeError(
                "\n" + "="*80 + "\n"
                "ERROR: Failed to install 'pyaudio'.\n"
                "This package requires the system-level 'portaudio' library to compile.\n"
                "Please install 'portaudio' using your package manager and try again:\n"
                "  - Ubuntu/Debian: sudo apt install portaudio19-dev\n"
                "  - macOS: brew install portaudio\n"
                "  - Fedora/RHEL: sudo dnf install portaudio-devel\n"
                "  - Conda: conda install pyaudio\n"
                "="*80 + "\n"
            )

# Step 4: Install pyfluidsynth
try:
    import fluidsynth
except ImportError:
    if not install_packages(["pyfluidsynth>=1.2.5"]):
        raise RuntimeError("ERROR: Failed to pre-install dependency 'pyfluidsynth'.")

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
              'checkpoints/beat/beat_blstm/variables/*',
              'checkpoints/chord/chord_v1/configurations.yaml',
              'checkpoints/chord/chord_v1/saved_model.pb',
              'checkpoints/chord/chord_v1/variables/*',
              'checkpoints/drum/drum_keras/*',
              'checkpoints/drum/drum_keras/variables/*',
              'checkpoints/music/music_note_stream/*',
              'checkpoints/music/music_note_stream/variables/*',
              'checkpoints/music/music_piano-v2/*',
              'checkpoints/music/music_piano-v2/variables/*',
              'checkpoints/music/music_piano/*',
              'checkpoints/music/music_piano/variables/*',
              'checkpoints/music/music_pop/*',
              'checkpoints/music/music_pop/variables/*',
              'checkpoints/patch_cnn/patch_cnn_melody/*',
              'checkpoints/patch_cnn/patch_cnn_melody/variables/*',
              'checkpoints/vocal/vocal_contour/*',
              'checkpoints/vocal/vocal_contour/variables/*',
              'checkpoints/vocal/vocal_semi/*',
              'checkpoints/vocal/vocal_semi/variables/*',
              'defaults/*',
              'resource/vamp/*']}

install_requires = \
['click>=7.1.2',
 'jsonschema>=3.2.0',
 'librosa>=0.8.0',
 'madmom>=0.16.1',
 'mir_eval>=0.6',
 'pillow>=8.3.2',
 'pretty_midi>=0.2.9',
 'pyfluidsynth>=1.2.5',
 'pyyaml>=5.3.1',
 'tensorflow>=2.5.0; python_version < "3.14"',
 'tf-nightly; python_version >= "3.14"',
 'tf-keras; python_version >= "3.9" and python_version < "3.14"',
 'tf-keras-nightly; python_version >= "3.14"',
 'setuptools < 82',
 'urllib3>=1.26.4',
 'vamp>=1.1.0']

extras_require = \
{'vocal': ['spleeter>=2.3.0']}

entry_points = \
{'console_scripts': ['omnizart = omnizart.cli.cli:entry']}

LONG_DESCRIPTION = open("README.md", "r", encoding="utf-8").read()

setup_kwargs = {
    'name': 'omnizart',
    'version': '0.5.0',
    'description': 'Omniscient Mozart, being able to transcribe everything in the music.',
    'long_description': LONG_DESCRIPTION,
    'author': 'BreezeWhite',
    'author_email': 'miyashita2010@tuta.io',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://sites.google.com/view/mctl/home',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'entry_points': entry_points,
    'python_requires': '>=3.8',
}


# Filter install_requires to exclude packages we have already pre-installed in this setup run.
# We verify if they are installed in the target environment using clean subprocess metadata queries on the target Python interpreter.
pre_installed_successfully = []

def is_target_installed(pkg_name):
    try:
        clean_env = os.environ.copy()
        clean_env.pop("PYTHONPATH", None)
        clean_env.pop("PYTHONNOUSERSITE", None)
        
        # Use importlib.metadata (standard in Python 3.8+) to check installation without importing the module itself,
        # which avoids triggering module-level imports of uninstalled runtime dependencies like scipy.
        subprocess.check_call(
            [sys.executable, "-c", f"import importlib.metadata; importlib.metadata.version('{pkg_name}')"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=clean_env
        )
        return True
    except Exception:
        try:
            subprocess.check_call(
                [sys.executable, "-c", f"import pkg_resources; pkg_resources.get_distribution('{pkg_name}')"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=clean_env
            )
            return True
        except Exception:
            return False

if is_target_installed("madmom"):
    pre_installed_successfully.append("madmom")
if is_target_installed("vamp"):
    pre_installed_successfully.append("vamp")
if is_target_installed("pyfluidsynth"):
    pre_installed_successfully.append("pyfluidsynth")

filtered_install_requires = []
for req in install_requires:
    is_pre_installed = False
    for pkg in pre_installed_successfully:
        if req.startswith(pkg):
            is_pre_installed = True
            break
    if not is_pre_installed:
        filtered_install_requires.append(req)

setup_kwargs['install_requires'] = filtered_install_requires

setup(**setup_kwargs)
