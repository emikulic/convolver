# Use plot.sh to drive this.
set logscale y
plot '/tmp/deconv' using 1:2 with lines title 'total', \
     '/tmp/deconv' using 1:3 with lines title 'diff'
pause -1
