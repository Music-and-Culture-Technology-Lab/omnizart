
# Development
Describes the neccessary background of how to develop this project.

## Download and install
``` bash
git clone https://github.com/Music-and-Culture-Technology-Lab/omnizart.git

# Install dependenies. For more different installation approaches, please refer to the official documentation page.
# The following command will download the checkpoints automatically.
cd omnizart
make install

# For developers, you have to install Dev dependencies as well, since they will not be installed by default.
make install-dev
```

## Package management
Uses [poetry](https://python-poetry.org/) for package management, instead of writing `requirements.txt` and `setup.py` manually.
We still provide the above two files for convenience. You can also generate them by executing ``make export``.

### ATTENTION! MUST SEE!
There is a major difference between install with `poetry install` and `python setup.py install`. When using poetry for installation, which
is the default approach when running `make insatll`, the site-pacakges and resource files are placed in the **current** folder.
This is different from executing `python setup.py install`, which resource files are installed in where a normal package you download through `pip install` will be placed (e.g. ~/.local/lib/python3.6/site-packages) .

And why things aren't placed in the normal path, but the command still can be executed? The answer is that poetry add an additional package path to your *PATH*  environment variable, and guess what is that path? Bingo! Your current path where you can execute `poetry install`! The difference has a major impact on running `omnizart download-checkpoints`. The default save path of checkpoints is to where omnizart being installed. That would be fine for end users, but not good news for developers though. That means after you git clone this project, and installed with `setup.py` approach, the **checkpoints are stored under ~/.local/.../site-packages/**, not your current development path. Therefore, it is strongly suggested that developers should use the default installation approach for a more comfortable developing experience^^.

Feedback: what a big trap there is...


## Documentation
Automatically generate documents from inline docstrings of module, class, and function. 
[Hosted document page](https://music-and-culture-technology-lab.github.io/omnizart-doc/index.html)

Documentation style: Follows `numpy` document flavor. Learn more from [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).

Document builder: [sphinx](https://www.sphinx-doc.org/en/master/)

To generate documents, `cd docs/` and execute `make html`. To see the rendered results, run `make serve` and view from the browser.
All documents and docstrings use **reStructured Text** format. More informations about this format can be found from 
[Sphinx's Document](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).

## Linters
Uses flake8 and pylint for coding style check.

To check with linters, execute `make check`.

You don't have to achieve a perfect score on pylint check, just pass 9.5 points still counted as a successful check.

### Caution!
There is convenient make command for formating the code, but it should be used very carefully.
Not only it could format the code for you, but also could mess up the code, and eventually you should still need
to check the format manually after refacorting the code with tools. 

To format the code with black and yapf, enter `make format`.

## Unittest
Uses `pytest` for unittest. The overall coverage rate should pass 25%, or CI would fail.

## CI/CD
Uses github actions for automatic linting, unittesting, document building, and package release.
Currently supports two workflows:
* General check
* Documentation page publishing
* Publish PyPI package and docker image

### General Check
Everytime you push to the master branch, file a pull request, and merge into master branch, will trigger
this action. This will do checks like code format, and unittests by leveraging the above mentioned
tools. If the check fails, you will not be able to merge the feature branch into master branch.

### Documentation Page Publishing
We use [github page](https://pages.github.com/) to host our documentation, and is separated as an [independent
repository](https://github.com/Music-and-Culture-Technology-Lab/omnizart-doc). 

**Please do not directly modify the content of the omnizart-doc repository!!**

The only permitted way to update the documentation page is by updating the `build_doc` branch, and
let the workflow do the rest of things.

Steps to update the documentation page:
* Clone **this** repo
* Create a new branch. **DO NOT UPDATE THE `build_doc` BRANCH DIRECTLY!!**
* File a pull request
* Merge into master (by admin)
* Merge into `build_doc` branch (by admin)
* Push to this repo (by admin)

### Publish PyPI Package and Docker Image
Publish the python package to PyPI and also the docker image to dockerhub when push tags to the repository.
The publish process will be automatically done by the github actions. There are several steps in the process:

1. Pack and publish the python package.
2. Build the docker image and publish to Docker Hub.
3. Create release -> this will also trigger the automation of documentation publishment.


## Docker
We provide both the Dockerfile for local image build and also the pre-build image on Docker Hub.

To build the image, run the following:
```
docker build -t omnizart:my-image .
```

To use the pre-build image, follow below steps:
```
# Download from Docker Hub
docker pull mctlab/omnizart

# Execute the image
docker run -it mctlab/omnizart:latest

### For those who want to leverage the power of GPU for acceleration, make sure
### you have installed docker>=19.03 and the 'nvidia-container-toolkit' package.
# Execute the docker with GPU support
docker run --gpus all -it mctlab/omnizart:latest
```


## Command Test
To actually install and test the `omnizart` command, execute `make install`. This will automatically create a virtual environment and install everything needed inside it. After installation, just follow the instruction showing on the screen to activate the environment, then type `omnizart --help` to check if it works. After testing the command, type `deactivate` to leave the virtual environment.

## Others
### Log Level
The default log level is set to `warn`. You can change it by exporting environment variable *LOG_LEVEL* to one of `debug`, `info`, `warning`, `error`, or `critical`. The verbosity is sorted from high to low (debug -> critical). For the consideration behind the log level design, please refer to the [soruce code](https://github.com/Music-and-Culture-Technology-Lab/omnizart/blob/master/omnizart/utils.py#L20) or the [documentation page](https://music-and-culture-technology-lab.github.io/omnizart-doc/utils.html#omnizart.utils.get_logger)
