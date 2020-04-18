FROM gitpod/workspace-full

USER root

RUN apt-get update \
 && apt-get install -y \
  apt-utils \
  sudo \
  git \
  less \
  wget

RUN mkdir -p /workspace/data \
    && chown -R gitpod:gitpod /workspace/data

USER gitpod

RUN mkdir /home/gitpod/.conda
# Install conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ~/conda && \
    rm ~/miniconda.sh && \
    conda init bash && \
    conda config --set auto_activate_base false

RUN conda activate base && \
    conda env create -f /workspace/astropop/.rtd-environment.yml && \
    conda clean -a

# Give back control
USER root

# Cleaning
RUN apt-get clean