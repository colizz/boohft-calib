#!/bin/bash

WORKDIR=$PWD
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# trick for SWAN: unset previous python env
unset PYTHONPATH
unset PYTHONHOME
# load CMSSW environment
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=${SCRAM_ARCH:-el9_amd64_gcc12}
export RELEASE=${BOOHFT_CMSSW_RELEASE:-CMSSW_14_1_9_patch2}
if [ -r $RELEASE/src ] ; then
  echo found $RELEASE
else
  echo please setup $RELEASE env first
  exit 1
fi
cd $RELEASE/src
eval `scram runtime -sh`

# launch the fit
cd $WORKDIR
python3 "$SCRIPT_DIR/fit.py" "$@"
