#!/bin/bash

WORKDIR=$PWD

# load CMSSW environment
source /cvmfs/cms.cern.ch/cmsset_default.sh
export RELEASE=CMSSW_10_2_27
if [ -r $RELEASE/src ] ; then
  echo release $RELEASE already exists
else
  source $WORKDIR/cmssw/env_setup.sh
fi
cd $RELEASE/src
eval `scram runtime -sh`

# launch the fit
cd $WORKDIR
python $WORKDIR/cmssw/fit.py $@