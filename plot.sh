#!/bin/bash
# Given multiple kernel dir, plots loss over steps.
if [[ $# -lt 1 ]]; then
  echo "usage: $0 kern-dirs..." >&2
  exit 1
fi
(
  echo "set logscale y"
  echo "plot \\"
  for i in $*; do
    fn="$i/log.txt"
    if [[ -e "$fn" ]]; then
      echo "  '$fn' using 6:8 with lines title '$i', \\"
    fi
  done
  echo ""
  echo "pause mouse close"
) | gnuplot
