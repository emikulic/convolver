#!/usr/bin/env python3
"""
Resizes a convolution kernel, by adding random fill around the border.
"""
import numpy as np
import os
os.environ['NO_TF'] = '1'
import argparse
import util

def main():
  p = argparse.ArgumentParser(description =
      'Given two images, determine the convolution kernel so that '
      'a * k = b')
  p.add_argument('ka', help='input kernel')
  p.add_argument('kb', help='output kernel directory')
  p.add_argument('n', type=int, help='output kernel size')
  p.add_argument('-mul', type=float, default=.5,
      help='multiplier for random fill')
  p.add_argument('-norm', type=float, default=0,
      help='normalize sum to this amount (default: zero: no normalization)')
  args = p.parse_args()

  os.mkdir(args.kb)
  step, kernel = util.load_kernel(args.ka)

  print('input kernel size', kernel.shape)
  e = np.mean(np.abs(kernel))
  print('input kernel mean', e)

  na = kernel.shape[0]
  nb = args.n

  outk = np.random.normal(size=(nb, nb, 1, 1)).astype(np.float32)
  oute = np.mean(np.abs(outk))
  outk *= e * args.mul / oute

  if nb > na:
    h = (nb - na) // 2
    print('grow: offset', h)
    outk[h:h+na, h:h+na, :, :] = kernel
  else:
    h = (na - nb) // 2
    print('shrink: offset', h)
    outk = kernel[h:h+nb, h:h+nb, :, :]

  if args.norm != 0:
    outk = outk / np.sum(outk) * args.norm

  oute = np.mean(np.abs(outk))
  print('output kernel mean', oute)

  util.save_kernel(args.kb, step, outk)
  print('resized from', kernel.shape, 'to', outk.shape)

if __name__ == '__main__':
  main()

# vim:set ts=2 sw=2 sts=2 et:
