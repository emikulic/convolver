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
  p.add_argument('-sym', type=boolchoice, default=True,
      choices=[True, False],
      help='kernel will be symmetric if set to True (default: True)')
  p.add_argument('-gamma', type=float, default=1.0,
      help='gamma correction to use for images (default: no correction)')
  p.add_argument('-learn_rate', type=float, default=2.**-10,
      help='learning rate for the optimizer')
  p.add_argument('-max_steps', type=int, default=0,
      help='stop after this many steps (default: zero: never stop)')
  p.add_argument('-num_steps', type=int, default=10,
      help='run this many optimizer steps at a time')
  #TODO
  #p.add_argument('-fps', type=float, default=5,
  #    help='how often to update the viewer, set to zero to disable viewer')
  args = p.parse_args()

  if not os.path.exists(args.k):
    os.mkdir(args.k)
    step = -1
  else:
    step, w1 = util.load_kernel(args.k)
    args.n = w1.shape[0]

  if step >= args.max_steps and args.max_steps != 0:
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
    # TODO rename w1
    log.log('Loaded weights.')
  else:
    assert step == -1, step
    step = 0
    log.log('Starting with random weights.')
    w1 = np.random.normal(size=(args.n, args.n, 1, 1),
        scale=.2).astype(np.float32)
    m = args.n // 2
    w1[m,m,0,0] = 1.  # Bright middle pixel.
  if args.sym:
    w1 = util.make_symmetric(w1)
  else:
    w1 = tf.Variable(w1)
  log.log('Weights shape is', w1.shape)

  # Build convolution model.
  model = keras.Sequential([
    keras.layers.DepthwiseConv2D(
      input_shape=img1.shape[1:],
      kernel_size=w1.shape[0],
      padding='valid',
      use_bias=False,
      kernel_regularizer=keras.regularizers.l1(0.000001), # todo: make tunable
      weights=[w1],
      )
  ])

  # Remove pixels outside of convolution output.
  _,ih,iw,_ = img1.shape
  _,oh,ow,_ = model.output.shape
  assert (ih - oh) % 2 == 0, (ih, oh)
  assert (iw - ow) % 2 == 0, (iw, ow)
  bh = (ih - oh) // 2
  bw = (iw - ow) // 2
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

  last_loss = None
  try:
    while True:
      fit = model.fit(input_img, expected_img, epochs=args.num_steps, verbose=0)
      step += fit.epoch[-1] + 1
      loss = fit.history['loss'][-1]
      saved = ''
      if last_loss is None or loss < last_loss:
        last_loss = loss
        saved = ' (saved)'
        util.save_kernel(args.k, step, model.layers[0].weights[0].numpy())
      log.log(f'step {step} loss {loss:.9f}{saved}')
      if args.max_steps > 0 and step >= args.max_steps:
        log.log(f'Hit max_steps={args.max_steps}')
        break
  except KeyboardInterrupt:
    pass
  log.log('Stopped.')
  log.close()

def delete_me():
  if False:
    rstep, rcost, rreg, rdiffcost = sess.run([
      global_step, cost, reg, diffcost])
    log.log('steps', rstep,
        'total-cost %.9f' % rcost,
        'diffcost %.9f' % rdiffcost,
        'reg %.9f' % rreg,
        'avg-px-err %.6f' % util.avg_px_err(rdiffcost, args.gamma),
        )
  render_time = [0.]
  def render():
    """Returns an image showing the current weights and output."""
    # TODO: vertically align labels.
    t0 = time.time()
    rout, rdiff, rw = sess.run([actual_img, diff, w1])
    render_out = util.vstack([
      util.hstack([
        util.vstack([util.cache_label('input:'), vimg1], 5),
        util.vstack([util.cache_label('actual:'),
          util.vis_nhwc(rout, doubles=0, gamma=args.gamma)], 5),
        util.vstack([util.cache_label('expected:'), vimg2], 5),
        ], 5),
      util.cache_label('difference:'), util.vis_nhwc(rdiff, doubles=0),
      util.cache_label('kernel:'), util.vis_hwoi(rw, doubles=2),
      ], 5)
    render_out = util.border(render_out, 5)
    t1 = time.time()
    render_time[0] += t1 - t0
    return render_out

  def periodic_save():
    rstep, rdiffcost, rw = sess.run([global_step, diffcost, w1])
    util.save_kernel(args.k, rstep, rw)
    rfn = args.k+'/render-step%08d-diff%.9f.png' % (rstep, rdiffcost)
    util.save_image(rfn, render())

if __name__ == '__main__':
  main()

# vim:set ts=2 sw=2 sts=2 et:
