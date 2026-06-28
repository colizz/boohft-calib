#!/bin/bash
set -e

SCRIPT_PATH=$(readlink -f "$1")
shift

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

unset PYTHONHOME
unset PYTHONPATH

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=${SCRAM_ARCH:-el9_amd64_gcc12}
export RELEASE=${BOOHFT_CMSSW_RELEASE:-CMSSW_14_1_9_patch2}
export RELEASE_DIR=${BOOHFT_CMSSW_RELEASE_DIR:-$ROUTINE_DIR/$RELEASE}

if [ ! -r "$RELEASE_DIR/src" ] ; then
    echo "CMSSW release $RELEASE_DIR is missing. Run routines/sfbdt/cmssw/env_setup.sh first."
    exit 1
fi

WORKDIR=$PWD
cd "$RELEASE_DIR/src"
eval `scram runtime -sh`
cd "$WORKDIR"

bash "$SCRIPT_PATH" "$@"
