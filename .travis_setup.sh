#!/bin/bash -x
hash -r
set -e

# Based on astropy ci-helpers

export PYTHONIOENCODING=UTF8
export LATEST_ASTROPY_STABLE=4.0
export LATEST_NUMPY_STABLE=1.17
MKL='nomkl'
is_number='[0-9]'
is_eq_number='=[0-9]'
is_eq_float="=[0-9]+\.[0-9]+"

if [ -z "$RETRY_ERRORS" ]; then
    RETRY_ERRORS="CondaHTTPError" # add more errors if needed (space-separated)
fi
if [ -z "$RETRY_MAX" ]; then
    RETRY_MAX=3
fi
if [ -z "$RETRY_DELAY" ]; then
    RETRY_DELAY=2
fi

# A wrapper for calls that should be repeated if their output contains any of
# the strings in RETRY_ERRORS.
##############################################################################
# CAUTION: This function will *unify* stdout and stderr of the wrapped call: #
#          In case of success, the call's entire output will go to stdout.   #
#          In case of failure, the call's entire output will go to stderr.   #
##############################################################################
function retry_on_known_error() {
    if [ -z "$*" ]; then
        echo "ERROR: Function retry_on_known_error() called without arguments." 1>&2
        return 1
    fi
    _tmp_output_file="tmp.txt"
    _n_retries=0
    _exitval=0
    _retry=true
    while $_retry; do
        _retry=false
        # Execute the wrapped command and get its unified output.
        # This command needs to run in the current shell/environment in case
        # it sets environment variables (like 'conda install' does)
        #
        # tee will both echo output to stdout and save it in a file. The file
        # is needed for some error checks later. Output to stdout is needed in
        # the event a conda solve takes a really long time (>10 min). If
        # there is no output on travis for that long, the job is cancelled.
        set +e
        $@ > $_tmp_output_file 2>&1
        _exitval="$?"
        # Keep the cat here...otherwise _exitval is always 0
        # even if the conda install failed.
        cat $_tmp_output_file
        set -e

        # The hack below is to work around a bug in conda 4.7 in which a spec
        # pinned in a pin file is not respected if that package is listed
        # explicitly on the command line even if there is no version spec on
        # the command line. See:
        #
        #   https://github.com/conda/conda/issues/9052
        #
        # The hacky workaround is to identify overridden specs and add the
        # spec from the pin file back to the command line.
        if [[ -n $(grep "conflicts with explicit specs" $_tmp_output_file) ]]; then
            # Roll back the command than generated the conflict message.
            # To do this, we get the most recent environment revision number,
            # then roll back to the one before that.
            # To ensure we don't need to activate, direct output of conda to
            # a file instead of piping
            _revision_file="revisions.txt"
            conda list --revision > _revision_file
            _current_revision=$(cat _revision_file | grep \(rev | tail -1 | cut -d' ' -f5 | cut -d')' -f1)
            conda install --revision=$(( $_current_revision - 1 ))
            _tmp_spec_conflicts=bad_spec.txt
            # Isolate the problematic specs
            grep "conflicts with explicit specs" $_tmp_output_file > $_tmp_spec_conflicts

            # Do NOT turn the three lines below into one by putting the python in a
            # $()...we need to make sure we stay in the shell in which conda is activated,
            # not a subshell.
            _tmp_updated_conda_command=new_command.txt
            python ci-helpers/travis/hack_version_numbers.py $_tmp_spec_conflicts "$@" > $_tmp_updated_conda_command
            revised_command=$(cat $_tmp_updated_conda_command)
            echo $revised_command
            # Try it; if it still has conflicts then just give up
            $revised_command > $_tmp_output_file 2>&1
            _exitval="$?"
            # Keep the cat here...otherwise _exitval is always 0
            # even if the conda install failed.
            cat $_tmp_output_file
            if [[ -n $(grep "conflicts with explicit specs" $_tmp_output_file) ]]; then
                echo "STOPPING conda attempts because unable to resolve conda pinning issues"
                rm -f $_tmp_output_file
                return 1
            fi
        fi

        # If the command was sucessful, abort the retry loop:
        if [ "$_exitval" == "0" ]; then
            break
        fi

        # The command errored, so let's check its output for the specified error
        # strings:
        if [[ $_n_retries -lt $RETRY_MAX ]]; then
            # If a known error string was found, throw a warning and wait a
            # certain number of seconds before invoking the command again:
            for _error in $RETRY_ERRORS; do
                if [ -n "$(grep "$_error" "$_tmp_output_file")" ]; then
                    echo "WARNING: The comand \"$@\" failed due to a $_error, retrying." 1>&2
                    _n_retries=$(($_n_retries + 1))
                    _retry=true
                    sleep $RETRY_DELAY
                    break
                fi
            done
        fi
    done
    # remove the temporary output file
    rm -f "$_tmp_output_file"
    # Finally, return the command's exit code:
    return $_exitval
}


echo "Installing Miniconda"
wget https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh --progress=dot:mega
mkdir $HOME/.conda
bash miniconda.sh -b -p $HOME/miniconda
$HOME/miniconda/bin/conda init bash
source ~/.bash_profile
conda activate base

echo "Install Conda Channels"
if [[ ! -z $CONDA_CHANNELS ]]; then
for channel in $CONDA_CHANNELS; do
    conda config --add channels $channel
done
fi
unset CONDA_CHANNELS

conda config --set always_yes yes --set changeps1 no
shopt -s nocasematch

echo "Install Environment"
retry_on_known_error conda install $QUIET conda

if [[ -z $PYTHON_VERSION ]]; then
    export PYTHON_VERSION=$TRAVIS_PYTHON_VERSION
fi
if [[ $NUMPY_VERSION == stable ]]; then
    export NUMPY_VERSION=$LATEST_NUMPY_STABLE
fi
if [[ $ASTROPY_VERSION == stable ]]; then
    export ASTROPY_VERSION=$LATEST_ASTROPY_STABLE
fi
if [[ $SCIPY_VERSION == stable ]]; then
    export ASTROPY_VERSION=$LATEST_SCIPY_STABLE
fi

echo "Using Python $PYTHON_VERSION"
echo "Using Astropy $ASTROPY_VERSION"
echo "Using Numpy $NUMPY_VERSION"
echo "Using Scipy $SCIPY_VERSION"

retry_on_known_error conda env create -n test -f $CONDA_ENVIRONMENT python=$PYTHON_VERSION numpy=$NUMPY_VERSION $MKL scipy=$SCIPY_VERSION astropy=$ASTROPY_VERSION

echo "Installing dependencies"
if [[ $MAIN_CMD == pep8* ]]; then
    $PIP_INSTALL pep8
    return  # no more dependencies needed
fi

if [[ $MAIN_CMD == pycodestyle* ]]; then
    $PIP_INSTALL pycodestyle
    return  # no more dependencies needed
fi

if [[ $MAIN_CMD == flake8* ]]; then
    $PIP_INSTALL flake8
    return  # no more dependencies needed
fi

if [[ $MAIN_CMD == pylint* ]]; then
    $PIP_INSTALL pylint
    return  # no more dependencies needed
fi

# Setting the MPL backend to a default to avoid occational segfaults with the qt backend
if [[ -z $MPLBACKEND ]]; then
    export MPLBACKEND=Agg
fi

echo "Final conda environments"
conda info -a
set +ex
