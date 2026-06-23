#!/bin/bash
set -e

WORKDIR=$PWD
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# trick for SWAN: unset previous python env
unset PYTHONPATH
unset PYTHONHOME
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=${SCRAM_ARCH:-el9_amd64_gcc12}
export RELEASE=${BOOHFT_CMSSW_RELEASE:-CMSSW_14_1_9_patch2}
export COMBINE_TAG=${BOOHFT_COMBINE_TAG:-v10.6.0}
export BUILD_CORES=${BOOHFT_CMSSW_BUILD_CORES:-8}

if [ -r $RELEASE/src ] ; then
    echo release $RELEASE already exists
else
    scram p CMSSW $RELEASE
fi

cd $RELEASE/src
eval `scram runtime -sh`

needs_build=0

if [ -d HiggsAnalysis/CombinedLimit/.git ] ; then
    echo HiggsAnalysis/CombinedLimit already exists
else
    git -c advice.detachedHead=false clone --depth 1 --branch $COMBINE_TAG https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
    needs_build=1
fi

if [ -d CombineHarvester/.git ] ; then
    echo CombineHarvester already exists
else
    git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
    needs_build=1
fi

if ! cmp -s "$SCRIPT_DIR/data/TagAndProbeExtendedV2.py" HiggsAnalysis/CombinedLimit/python/TagAndProbeExtendedV2.py ; then
    cp "$SCRIPT_DIR/data/TagAndProbeExtendedV2.py" HiggsAnalysis/CombinedLimit/python/
    needs_build=1
fi

if ! cmp -s "$SCRIPT_DIR/data/plot1DScanWithOutput.py" CombineHarvester/CombineTools/scripts/plot1DScanWithOutput.py ; then
    cp "$SCRIPT_DIR/data/plot1DScanWithOutput.py" CombineHarvester/CombineTools/scripts/
    needs_build=1
fi

if [ $needs_build -eq 1 ] || ! command -v combine >/dev/null 2>&1 ; then
    scramv1 b -j$BUILD_CORES
else
    echo "$RELEASE with Combine $COMBINE_TAG is already usable"
fi
