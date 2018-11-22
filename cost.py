#!/usr/bin/env python3
"""
Given two images and some kernels, report the difference per kernel.
"""
import time
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import util

def main():
  p = argparse.ArgumentParser(description =
      'Given two images and some kernels, report the difference per kernel.')
  p.add_argument('a', help='input image filename')
  p.add_argument('b', help='expected image filename')
  p.add_argument('kernels', nargs='*', help='kernel directory')
  p.add_argument('-gamma', type=float, default=1.0,
      help='gamma correction to use for images (default: no correction)')
  p.add_argument('-crop_x', type=int, default=0,
      help='crop X offset in pixels, range is [0..width-1]')
  p.add_argument('-crop_y', type=int, default=0,
      help='crop Y offset in pixels, range is [0..height-1] where 0 is the TOP')
  p.add_argument('-crop_w', type=int, default=0,
      help='crop width in pixels')
  p.add_argument('-crop_h', type=int, default=0,
      help='crop height in pixels')
  args = p.parse_args()

  img1 = util.load_image(args.a, args)
  img2 = util.load_image(args.b, args)
  assert img1.shape == img2.shape, (img1.shape, img2.shape)
  print('# Loaded images. Shape is', img1.shape)

  img_input = tf.constant(img1)
  img_expected = tf.constant(img2)
  sess = util.make_session()

  for kfn in args.kernels:
    step, kernel = util.load_kernel(kfn)
    n = kernel.shape[0]
    border = (n + 1) // 2

    # Convolve and calculate costs.
    img_actual = util.convolve(img_input, kernel)
    dcost = sess.run(util.diff_cost(
      util.diff(img_actual, img_expected, border)))
    rcost = sess.run(util.reg_cost(kernel))

    print(kfn, 'n', n, 'diffcost %.12f' % dcost, 'regcost', rcost,
        'avg-px-err', util.avg_px_err(dcost, args.gamma))

if __name__ == '__main__':
  main()

# vim:set ts=2 sw=2 sts=2 et:
