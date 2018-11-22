#!/bin/bash
# Given a kernel dir, plots the costs over time.
if [[ $# != 1 ]]; then
  echo "usage: $0 kern-dir" >&2
  exit 1
fi
DATA=/tmp/deconv
grep total-cost $1/log.txt | awk '{print $6, $8, $10}' >$DATA
gnuplot plot.gnuplot
rm -f $DATA
