#!/usr/bin/env python3
"""
Convolve an image with a kernel.
"""
import tensorflow as tf
import numpy as np
import os
import argparse
import util

def main():
  p = argparse.ArgumentParser(description = 'Convolve an image with a kernel.')
  p.add_argument('img', help='input image filename')
  p.add_argument('k', help='path to kernel ')
  p.add_argument('-gamma', type=float, default=1.0,
      help='gamma correction to use for images (default: no correction)')
  p.add_argument('-out', help='output to *.png file instead of viewing')
  args = p.parse_args()

  img = util.load_image(args.img, args, gray=False)
  n,h,w,c = img.shape
  assert n == 1, n
  out = np.zeros((h,w,c), dtype=np.float32)
  step, kernel = util.load_kernel(args.k)
  util.tf_init()
  for i in range(c):
    chan = util.convolve(tf.constant(img[:,:,:,i:i+1]), kernel)
    out[:,:,i] = chan[0,:,:,0]
  out = util.from_float(out, args.gamma)

  if args.out is not None:
    util.save_image(args.out, out)
    print('Written to', args.out)
  else:
    print('Press ESC to close window.')
    util.viewer(None, lambda: out)

if __name__ == '__main__':
  main()

# vim:set ts=2 sw=2 sts=2 et:
