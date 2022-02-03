#!/bin/bash

WORKDIR=$PWD

# trick for SWAN: unset previous python env
unset PYTHONPATH
unset PYTHONHOME
# load CMSSW environment
source /cvmfs/cms.cern.ch/cmsset_default.sh
export RELEASE=CMSSW_10_2_27
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
python $WORKDIR/cmssw/fit.py $@