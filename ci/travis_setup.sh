#!/bin/bash -x

set -e

export NUMPY_STABLE="1.17"
export ASTROPY_STABLE="4.0"
export SCIPY_STABLE="1.4"
export MKL="nomkl"

MINICONDA="$HOME/miniconda"

# Install and update conda
echo "-----------------------------------------------"
echo "Setup Conda"
echo "-----------------------------------------------"
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p "$MINICONDA"
source "$MINICONDA/etc/profile.d/conda.sh"
conda config --set always_yes yes --set changeps1 no
conda update conda
hash -r

# Create test environment
echo "-----------------------------------------------"
echo "Creating environment"
echo "-----------------------------------------------"

conda config --add channels astropy
conda config --add channels juliotux
conda config --add channels conda-forge

if [[ -z $PYTHON_VERSION ]]; then
    echo "ERROR: Empty python version. Must be set in PYTHON_VERSION"
    return 1
else
    echo "Using python $PYTHON_VERSION"
    conda create -n $NAME -q python="$PYTHON_VERSION"
fi

echo "Entering in test environment"
conda activate $NAME
conda install -q pip
conda config --show
conda info -a
PIN_FILE=$MINICONDA/envs/$NAME/conda-meta/pinned
touch "$PIN_FILE"

if [[ $SETUP_CMD == egg_info ]]; then
    return  # no more dependencies needed
fi

echo "-----------------------------------------------"
echo "Installing Dependencies"
echo "-----------------------------------------------"

if [[ $MAIN_CMD == pep8* ]]; then
    pip install -U -q pep8
    return  # no more dependencies needed
elif [[ $MAIN_CMD == pycodestyle* ]]; then
    pip install -U -q pycodestyle
    return  # no more dependencies needed
elif [[ $MAIN_CMD == flake8* ]]; then
    pip install -U -q flake8
    return  # no more dependencies needed
elif [[ $MAIN_CMD == pylint* ]]; then
    pip install -U -q pylint
    return  # no more dependencies needed
fi

if [[ -z $NUMPY_VERSION ]]; then
    echo "Empty numpy version. Setting using default."
    conda install -n $NAME numpy "$MKL"
elif [[ $NUMPY_VERSION == stable ]]; then
    echo "Using stable numpy version. Set to $NUMPY_STABLE"
    conda install -n $NAME -q numpy="$NUMPY_STABLE" "$MKL"
elif [[ $NUMPY_VERSION == dev* ]] || [[ $NUMPY_VERSION == unstable ]]; then
    echo "Using development numpy. Installing from git."
    pip install git+https://github.com/numpy/numpy.git
else
    echo "Using numpy $NUMPY_VERSION"
    conda install -n $NAME -q numpy="$NUMPY_VERSION" "$MKL"
    echo "numpy=$NUMPY_VERSION.*" >> "$PIN_FILE"
fi

if [[ -z $SCIPY_VERSION ]]; then
    echo "Empty scipy version. Setting using default."
    conda install -n $NAME -q scipy
elif [[ $SCIPY_VERSION == stable ]]; then
    echo "Using stable scipy version. Set to $SCIPY_STABLE"
    conda install -n $NAME -q scipy=$SCIPY_STABLE
elif [[ $SCIPY_VERSION == dev* ]] || [[ $SCIPY_VERSION = unstable ]]; then
    echo "Using development scipy. Installing from git."
    pip install git+https://github.com/scipy/scipy.git
else
    echo "Using numpy $SCIPY_VERSION"
    conda install -n $NAME -q scipy="$SCIPY_VERSION"
    echo "scipy=$SCIPY_VERSION.*" >> "$PIN_FILE"
fi

if [[ -z $ASTROPY_VERSION ]]; then
    echo "Empty astropy version. Setting using default."
    conda install -n $NAME -q astropy
elif [[ $ASTROPY_VERSION == stable ]]; then
    echo "Using stable astropy version. Set to $ASTROPY_STABLE"
    conda install -n $NAME -q astropy=$ASTROPY_STABLE
elif [[ $ASTROPY_VERSION == dev* ]] || [[ $ASTROPY_VERSION == unstable ]]; then
    echo "Using development astropy. Installing from git."
    pip install git+https://github.com/astropy/astropy.git
else
    echo "Using astropy $ASTROPY_VERSION"
    conda install -n $NAME -q astropy="$ASTROPY_VERSION"
    echo "astropy=$ASTROPY_VERSION.*" >> "$PIN_FILE"
fi

if [[ $SETUP_CMD == *coverage* ]]; then
    # We install requests with conda since it's required by coveralls.
    echo "Installing coverage"
    conda install -n $NAME -q coverage requests
    pip install -U -q coveralls codecov codacy-coverage
fi

if [[ $SETUP_CMD == *-cov* ]]; then
    echo "Installing coverage"
    pip install -U -q coveralls codecov pytest-cov codacy-coverage
fi

if [[ $SETUP_CMD == *build_sphinx* ]] || [[ $SETUP_CMD == *build_docs* ]]; then
    if [[ ! -z $MATPLOTLIB_VERSION ]]; then
        if [[ -z $(grep matplotlib "$PIN_FILE") ]]; then
            echo "matplotlib ${MATPLOTLIB_VERSION}.*" >> "$PIN_FILE"
        fi
    fi
    
    if [[ ! -z $SPHINX_VERSION ]]; then
        if [[ -z $(grep sphinx "$PIN_FILE") ]]; then
            echo "sphinx ${SPHINX_VERSION}.*" >> "$PIN_FILE"
        fi
    fi

    conda install -q sphinx matplotlib sip
    
    # test matplotlib import
    python -c "import matplotlib.pyplot"
fi

# Another pins are ignored
echo "Installing another packages"
conda env update -f "$CONDA_ENVIRONMENT"


echo "-----------------------------------------------"
echo "Environment done."
echo "-----------------------------------------------"
conda list -n $NAME

set +ex
