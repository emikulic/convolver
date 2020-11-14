#!/usr/bin/env python3
"""
Given two images, determine the convolution kernel.
"""
import time
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import util
from tensorflow import keras

def boolchoice(s):
  if s == 'True': return True
  if s == 'False': return False
  assert False, s

def main():
  p = argparse.ArgumentParser(description =
      'Given two images, determine the convolution kernel so that '
      'a * k = b')
  p.add_argument('a', help='input image filename')
  p.add_argument('b', help='expected image filename')
  p.add_argument('k', help='kernel directory')
  p.add_argument('-n', type=int, default=5,
      help='kernel size is NxN (default: 5, or automatically set to size '
      'of loaded kernel)')
  # TODO: symmetry doesn't work yet.
  p.add_argument('-sym', type=boolchoice, default=True,
      choices=[True, False],
      help='kernel will be symmetric if set to True (default: True)')
  p.add_argument('-gamma', type=float, default=1.0,
      help='gamma correction to use for images (default: no correction)')
  p.add_argument('-reg_cost', type=float, default=0.00001,
      help='regularization cost: pressures the optimizer to minimize the '
      'weights in the kernel')
  p.add_argument('-learn_rate', type=float, default=2.**-10,
      help='learning rate for the optimizer')
  p.add_argument('-max_steps', type=int, default=0,
      help='stop after this many steps (default: zero: never stop)')
  p.add_argument('-num_steps', type=int, default=10,
      help='run this many optimizer steps at a time')
  p.add_argument('-crop_x', type=int, default=0,
      help='crop X offset in pixels, range is [0..width-1]')
  p.add_argument('-crop_y', type=int, default=0,
      help='crop Y offset in pixels, range is [0..height-1] where 0 is the TOP')
  p.add_argument('-crop_w', type=int, default=0,
      help='crop width in pixels')
  p.add_argument('-crop_h', type=int, default=0,
      help='crop height in pixels')
  p.add_argument('-fps', type=float, default=5,
      help='how often to update the viewer, set to zero to disable viewer')
  args = p.parse_args()

  if not os.path.exists(args.k):
    os.mkdir(args.k)
    step = -1
  else:
    step, weights = util.load_kernel(args.k)
    args.n = weights.shape[0]

  if step >= args.max_steps and args.max_steps > 0:
    print('Current step %d is over max %d. Exiting.' % (step, args.max_steps))
    return 0

  log = util.Logger(args.k + '/log.txt')
  log.log('--- Start of run ---')
  log.log('Cmdline:', sys.argv)

  # Load images.
  img1 = util.load_image(args.a, args)
  img2 = util.load_image(args.b, args)
  assert img1.shape == img2.shape, (img1.shape, img2.shape)
  log.log('Loaded images. Shape is', img1.shape, '(NHWC)')
  util.tf_init()

  vimg1 = util.vis_nhwc(img1, doubles=0, gamma=args.gamma)
  vimg2 = util.vis_nhwc(img2, doubles=0, gamma=args.gamma)

  # Load or initialize weights.
  if step >= 0:
    log.log('Loaded weights.')
  else:
    assert step == -1, step
    step = 0
    log.log('Starting with random weights.')
    weights = np.random.normal(size=(args.n, args.n, 1, 1),
        scale=.2).astype(np.float32)
    m = args.n // 2
    weights[m,m,0,0] = 1.  # Bright middle pixel.
  if args.sym:
    weights = util.make_symmetric(weights)
  else:
    weights = tf.Variable(weights)
  log.log('Weights shape is', weights.shape)

  # Build convolution model.
  model = keras.Sequential([
    keras.layers.DepthwiseConv2D(
      input_shape=img1.shape[1:],
      kernel_size=weights.shape[0],
      padding='valid',
      use_bias=False,
      kernel_regularizer=keras.regularizers.l1(args.reg_cost),
      weights=[weights],
      )
  ])
  del weights

  # Remove pixels outside of convolution output.
  _,ih,iw,_ = img1.shape
  _,oh,ow,_ = model.output.shape
  assert (ih - oh) % 2 == 0, (ih, oh)
  assert (iw - ow) % 2 == 0, (iw, ow)
  bh = (ih - oh) // 2
  bw = (iw - ow) // 2
  assert bw == bh, (bw, bh)
  img2 = img2[:, bh:-bh, bw:-bw, :]

  input_img = tf.constant(img1)
  expected_img = tf.constant(img2)

  # Optimizer.
  opt = keras.optimizers.Adam(lr=args.learn_rate)
  model.compile(
      optimizer=opt,
      loss='mean_squared_error',
  )
  model.summary()

  log.log('Current args:', args.__dict__)
  log.log('Starting at step', step)

  def render():
    """Returns an image showing the current weights and output."""
    # TODO: vertically align labels.
    rout = model(input_img).numpy()
    rdiff = img2 - rout
    rw = model.layers[0].weights[0].numpy()
    render_out = util.vstack([
      util.hstack([
        util.vstack([util.cache_label('input:'), vimg1], 5),
        util.vstack([util.cache_label('actual:'),
          util.border(
            util.vis_nhwc(rout, doubles=0, gamma=args.gamma), bw*2)], 5),
        util.vstack([util.cache_label('expected:'), vimg2], 5),
        ], 5),
      util.hstack([
        util.vstack([
          util.cache_label('difference:'),
          util.border(util.vis_nhwc(rdiff, doubles=0), bw*2),
        ], 5),
        util.vstack([
          util.cache_label('kernel:'), util.vis_hwoi(rw, doubles=2),
        ], 5),
      ], 5),
    ], 5)
    render_out = util.border(render_out, 5)
    return render_out

  if args.fps > 0:
    v = util.Viewer()
  last_loss = None
  try:
    while True:
      if args.fps > 0:
        if not v.running():
          log.log('Viewer closed.')
          break
        if v.passed(1. / args.fps):
          v.show(render())
      fit = model.fit(input_img, expected_img, epochs=args.num_steps, verbose=0)
      step += args.num_steps
      loss = fit.history['loss'][-1]
      saved = ''
      if last_loss is None or loss < last_loss:
        last_loss = loss
        saved = ' (saved)'
        util.save_kernel(args.k, step, model.layers[0].weights[0].numpy())
      log.log(f'step {step} loss {loss:.9f}{saved}')
      if step >= args.max_steps and args.max_steps > 0:
        log.log(f'Hit max_steps={args.max_steps}')
        break
  except KeyboardInterrupt:
    pass
  log.log('Stopped.')
  log.close()

if __name__ == '__main__':
  main()

# vim:set ts=2 sw=2 sts=2 et:
