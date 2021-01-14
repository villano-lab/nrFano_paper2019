#!/bin/sh

E=$1
n=$2

sed -e 's|ENERGY|'$E'|g' -e 's|NUM|'$n'|g' < condor-NRFano-sigpoints-E > condor_job

condor_submit condor_job
rm condor_job
