#!/bin/bash
set -e

SCRIPT_PATH=$(readlink -f "$1")
shift

unset PYTHONHOME
unset PYTHONPATH

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=${SCRAM_ARCH:-el9_amd64_gcc12}
export RELEASE=${BOOHFT_CMSSW_RELEASE:-CMSSW_14_1_9_patch2}

if [ ! -r "$RELEASE/src" ] ; then
    echo "CMSSW release $RELEASE is missing. Run env_setup.sh first."
    exit 1
fi

WORKDIR=$PWD
cd "$RELEASE/src"
eval `scram runtime -sh`
cd "$WORKDIR"

bash "$SCRIPT_PATH" "$@"
