#!/bin/bash

WORKDIR=$PWD

# trick for SWAN: unset previous python env
unset PYTHONPATH
unset PYTHONHOME
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc700
export RELEASE=CMSSW_10_2_27

if [ -r $RELEASE/src ] ; then
    echo release $RELEASE already exists
else
    scram p CMSSW $RELEASE
    cd $RELEASE/src
    eval `scram runtime -sh`

    ## Install HiggsAnalysis
    git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
    cp $WORKDIR/cmssw/data/TagAndProbeExtendedV2.py HiggsAnalysis/CombinedLimit/python/  # copy the model we will use in fit
    cd HiggsAnalysis/CombinedLimit
    git checkout v8.2.0 # recommended tag
    cd ../..

    ## Install CombineHarvester
    git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester -b 102x
    cp $WORKDIR/cmssw/data/plot1DScanWithOutput.py sc
    scram b -j8

    cd ../..
fi