#!/bin/bash
set -e

WORKDIR=$PWD
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# trick for SWAN: unset previous python env
unset PYTHONPATH
unset PYTHONHOME
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=${SCRAM_ARCH:-el9_amd64_gcc12}
export RELEASE=${BOOHFT_CMSSW_RELEASE:-CMSSW_14_1_9_patch2}
export RELEASE_DIR=${BOOHFT_CMSSW_RELEASE_DIR:-$ROUTINE_DIR/$RELEASE}
export COMBINE_TAG=${BOOHFT_COMBINE_TAG:-v10.6.0}
export BUILD_CORES=${BOOHFT_CMSSW_BUILD_CORES:-8}

cd "$ROUTINE_DIR"
if [ -r "$RELEASE_DIR/src" ] ; then
    echo release "$RELEASE_DIR" already exists
else
    scram p CMSSW "$RELEASE"
fi

cd "$RELEASE_DIR/src"
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

if ! cmp -s "$SCRIPT_DIR/data/TagAndProbeExtended.py" HiggsAnalysis/CombinedLimit/python/TagAndProbeExtended.py ; then
    cp "$SCRIPT_DIR/data/TagAndProbeExtended.py" HiggsAnalysis/CombinedLimit/python/
    needs_build=1
fi

if [ $needs_build -eq 1 ] || ! command -v combine >/dev/null 2>&1 ; then
    scramv1 b -j$BUILD_CORES
else
    echo "$RELEASE_DIR with Combine $COMBINE_TAG is already usable"
fi

cd "$WORKDIR"
