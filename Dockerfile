FROM tensorflow/tensorflow:2.3.1-gpu

WORKDIR /tmp

COPY omnizart ./omnizart
COPY scripts ./scripts
COPY Makefile ./
COPY pyproject.toml ./
COPY poetry.lock ./
COPY README.md ./

RUN apt-get update
RUN apt-get install --assume-yes libsndfile1 libgl1-mesa-glx ffmpeg vim
RUN scripts/install.sh

# Upgrade this for avoiding mysterious import module not found 'keyrings'
RUN pip install --upgrade keyrings.alt

WORKDIR /home
RUN mv /tmp/omnizart /usr/local/lib/python3.6/dist-packages
RUN rm -rf /tmp
COPY README.md ./

