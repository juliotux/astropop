#!/bin/bash

export NUMPY_STABLE=1.17
export ASTROPY_STABLE=4.0
export SCIPY_STABLE=1.4

# Install and update conda
echo "-----------------------------------------------"
echo "Setup Conda"
echo "-----------------------------------------------"
sudo apt-get update
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
source "$HOME/miniconda/etc/profile.d/conda.sh"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a

# Create test environment
echo "-----------------------------------------------"
echo "Creating environment"
echo "-----------------------------------------------"

if [[ ! -z $CONDA_CHANNELS ]]; then
    for channel in $CONDA_CHANNELS; do
        echo "Adding $channel channel"
        conda config --add channels $channel
    done
fi
unset CONDA_CHANNELS

if [[ -z $PYTHON_VERSION ]]; them
    echo "ERROR: Empty python version. Must be set in PYTHON_VERSION"
    return 1
else
    echo "Using python $PYTHON_VERSION"
    conda create -n test python=$PYTHON_VERSION
fi

echo "Entering in test environment"
conda activate test
conda config --show
conda info -a

PIN_FILE_CONDA=$HOME/miniconda/envs/test/conda-meta/pinned
touch $PIN_FILE

if [[ $SETUP_CMD == egg_info ]]; then
    return  # no more dependencies needed
fi

echo "-----------------------------------------------"
echo "Installing Dependencies"
echo "-----------------------------------------------"

if [[ -z $NUMPY_VERSION ]]; them
    echo "Empty numpy version. Setting using default."
    conda install -n test numpy
elif [[ $NUMPY_VERSION == stable ]]; them
    echo "Using stable numpy version. Set to $NUMPY_STABLE"
    conda install -n test scipy=$NUMPY_STABLE
elif [[ $NUMPY_VERSION == dev* ]] || [[ $NUMPY_VERSION == unstable]]; them
    echo "Using development numpy. Installing from git."
    pip install git+https://github.com/numpy/numpy.git
else
    echo "Using numpy $NUMPY_VERSION"
    conda install -n test numpy=$NUMPY_VERSION $MKL
    echo "numpy=$NUMPY_VERSION.*" >> $PIN_FILE
fi

if [[ -z $SCIPY_VERSION ]]; them
    echo "Empty scipy version. Setting using default."
    conda install -n test scipy
elif [[ $SCIPY_VERSION == stable ]]; them
    echo "Using stable scipy version. Set to $SCIPY_STABLE"
    conda install -n test scipy=$SCIPY_STABLE
elif [[ $SCIPY_VERSION == dev* ]] || [[ $SCIPY_VERSION == unstable]]; them
    echo "Using development scipy. Installing from git."
    pip install git+https://github.com/scipy/scipy.git
else
    echo "Using numpy $SCIPY_VERSION"
    conda install -n test scipy=$SCIPY_VERSION
    echo "numpy=$SCIPY_VERSION.*" >> $PIN_FILE
fi

if [[ -z $ASTROPY_VERSION ]]; them
    echo "Empty astropy version. Setting using default."
    conda install -n test astropy
elif [[ $ASTROPY_VERSION == stable ]]; them
    echo "Using stable astropy version. Set to $ASTROPY_STABLE"
    conda install -n test astropy=$ASTROPY_STABLE
elif [[ $ASTROPY_VERSION == dev* ]] || [[ $ASTROPY_VERSION == unstable]]; them
    echo "Using development astropy. Installing from git."
    pip install git+https://github.com/astropy/astropy.git
else
    echo "Using astropy $ASTROPY_VERSION"
    conda install -n test astropy=$ASTROPY_VERSION
    echo "numpy=$ASTROPY_VERSION.*" >> $PIN_FILE
fi

# Another pins are ignored
conda env update -n test -f ../rtd-environment.yml

echo "-----------------------------------------------"
echo "Environment done."
echo "-----------------------------------------------"
conda info -a