#!/bin/sh

EBIN=$1

list=`seq 50`
for i in ${list}
do
  STARTVAL=$(((i-1)*100))
  echo ${STARTVAL}
  sed -e 's|STARTIDX|'${STARTVAL}'|g' < condor-checkDifference-E${EBIN} > testcondor
  condor_submit testcondor
  sleep 5
  rm testcondor
done
