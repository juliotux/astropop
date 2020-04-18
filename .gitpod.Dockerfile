FROM gitpod/workspace-full

SHELL ["/bin/bash", "-c"]

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
    source ~/conda/etc/profile.d/conda.sh && \
    conda init --all --dry-run --verbose && \
    conda config --set auto_activate_base false

RUN source ~/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda create -n astropop python=3.7 numpy=1.17 && \
    conda install -n astropop -c juliotux -c astropy -c conda-forge \
                            astropy cython matplotlib \
                            sphinx-astropy pytz pyyaml \
                            sckikit-image scipy astroquery \
                            astroscrappy astroallign ccdproc \
                            photutils sep pytest-astropy \
                            pip ipython && \
    pip install sphinx-rtd-theme shinxcontrib-apidoc \
                astropy-helpers pytest_check \
                sphinxcontrib-napoleon && \
    conda clean -a

# Give back control
USER root

# Cleaning
RUN apt-get clean