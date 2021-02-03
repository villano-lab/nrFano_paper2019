#!/bin/sh

i=$1
n=$2

sed -e 's|IDX|'$i'|g' -e 's|NUM|'$n'|g' < condor-NRFano-sigpoints > condor_job

condor_submit condor_job
rm condor_job
