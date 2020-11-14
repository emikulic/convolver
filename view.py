#!/usr/bin/env python3
"""
Display one or more kernels.
"""
import os
os.environ['NO_TF'] = '1'
import numpy as np
import argparse
import util

def main():
  p = argparse.ArgumentParser(description =
      'Display a kernel.')
  p.add_argument('-out', help='output to *.png file instead of viewing')
  p.add_argument('k', nargs='+', help='path to kernel(s)')
  args = p.parse_args()

  out = None

  for fn in args.k:
    print('Loading', fn)
    step, kernel = util.load_kernel(fn)
    print('  Step', step)
    print('  Kernel shape is', kernel.shape)
    print('  Min', np.min(kernel))
    print('  Max', np.max(kernel))
    print('  Mean', np.mean(kernel))
    print('  Sum', np.sum(kernel))
    print('  Sum of abs', np.sum(np.abs(kernel)))
    print('  RMS', np.sqrt(np.mean(kernel * kernel)))

    render = util.vis_hwoi(kernel, doubles=2)
    render = util.hstack([render, util.make_label(fn)], 5)

    if out is None:
      out = render
    else:
      out = util.vstack([out, render], 5)

  out = util.border(out, 5)

  if args.out is not None:
    util.save_image(args.out, out)
    print('Written to', args.out)
  else:
    print('Press ESC to close window.')
    util.show(out)

if __name__ == '__main__':
  main()

# vim:set ts=2 sw=2 sts=2 et:
