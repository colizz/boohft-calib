#!/bin/bash

SCRIPT_PATH=$1

RUN_ON_SINGULARITY=0
if [ -f /etc/os-release ] && grep -q 'VERSION_ID="9' /etc/os-release; then
    echo "Running on EL9 machine, launching with singularity"
    RUN_ON_SINGULARITY=1
fi

if [ $RUN_ON_SINGULARITY -eq 0 ]; then
    bash $SCRIPT_PATH ${@:2}
else
    export SINGULARITY_CACHEDIR="/tmp/$(whoami)/singularity"
    singularity run -B /cvmfs -B /etc/grid-security -B /etc/pki/ca-trust --home $PWD:$PWD /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/cc7:x86_64 bash $SCRIPT_PATH ${@:2}
fi 
