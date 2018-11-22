#!/usr/bin/env python3
"""
Given two images, determine the convolution kernel.
"""
import time
INIT_T = time.time()
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import util

def boolchoice(s):
  if s == 'True': return True
  if s == 'False': return False
  assert False, s

def main():
  MAIN_T = time.time()
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
  p.add_argument('-reg_cost', type=float, default=0.,
      help='regularization cost: the sum of weights is multiplied by this '
      'and added to the cost (default: zero: no regularization)')
  p.add_argument('-border', type=int, default=-1,
      help='how many pixels to remove from the border (from every edge of the '
      'image) before calculating the difference (default: auto based on '
      'kernel size)')
  p.add_argument('-learn_rate', type=float, default=2.**-10,
      help='learning rate for the optimizer')
  p.add_argument('-epsilon', type=float, default=.09,
      help='epsilon for the optimizer')
  p.add_argument('-max_steps', type=int, default=0,
      help='stop after this many steps (default: zero: never stop)')
  p.add_argument('-log_every', type=int, default=100,
      help='log stats every N steps (0 to disable)')
  p.add_argument('-save_every', type=int, default=500,
      help='save kernel and image every N steps (0 to disable)')
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

  vimg1 = util.vis_nhwc(img1, doubles=0, gamma=args.gamma)
  vimg2 = util.vis_nhwc(img2, doubles=0, gamma=args.gamma)

  # Load and initialize weights.
  if step >= 0:
    log.log('Loaded weights, shape is', w1.shape, '(HWIO)')
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

  if args.border == -1:
    args.border = (args.n + 1) // 2
    log.log('Automatically set border to', args.border)

  log.log('Current args:', args.__dict__)
  log.log('Starting at step', step)

  # Convolution.
  input_img = tf.constant(img1)
  expected_img = tf.constant(img2)
  actual_img = util.convolve(input_img, w1)  # <-- THIS IS THE CALCULATION.

  # Cost.
  diff = util.diff(actual_img, expected_img, args.border)
  diffcost = util.diff_cost(diff)  # L2
  cost = diffcost

  # Regularization.
  reg = util.reg_cost(w1)  # L1
  if args.reg_cost != 0:
    cost += reg * args.reg_cost

  # Optimizer.
  global_step = tf.Variable(step, dtype=tf.int32, trainable=False,
      name='global_step')
  train_step = tf.train.AdamOptimizer(args.learn_rate, args.epsilon).minimize(
      cost, global_step=global_step)

  log.log('Starting TF session.')
  sess = util.make_session(outdir=args.k)

  # Get ready for viewer.
  log_last_step = [step]
  log_last_time = [time.time()]
  def periodic_log():
    """Does a log.log() of stats like step number and current error."""
    now = time.time()
    rstep, rcost, rreg, rdiffcost = sess.run([
      global_step, cost, reg, diffcost])
    if log_last_step[0] == rstep: return  # Dupe call.
    log.log('steps', rstep,
        'total-cost %.9f' % rcost,
        'diffcost %.9f' % rdiffcost,
        'reg %.9f' % rreg,
        'avg-px-err %.6f' % util.avg_px_err(rdiffcost, args.gamma),
        'steps/sec %.2f' % (
          (rstep - log_last_step[0]) / (now - log_last_time[0])),
        )
    log_last_step[0] = rstep
    log_last_time[0] = now

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

  calc_time = [0.]
  def calc_fn():
    """
    Run train_step.
    Then do every-N-steps housekeeping.
    """
    t0 = time.time()
    sess.run(train_step)  # <--- THIS IS WHERE THE MAGIC HAPPENS.
    t1 = time.time()
    calc_time[0] += t1 - t0
    nsteps = sess.run(global_step)
    if args.log_every != 0:
      if nsteps == 1 or nsteps % args.log_every == 0:
        periodic_log()
    if args.save_every != 0:
      if nsteps % args.save_every == 0:
        periodic_save()
    if args.max_steps == 0:
      return True  # Loop forever.
    return nsteps < args.max_steps

  log.log('Start optimizer.')
  START_T = time.time()
  if args.fps == 0:
    while True:
      if not calc_fn(): break
  else:
    util.viewer(calc_fn, render, fps=args.fps, hang=False)
  STOP_T = time.time()
  # Final log and save.
  log.log('Stop optimizer.')
  log.log('Render time %.3fs (%.02f%% of optimizer)' % (
    render_time[0], 100.*render_time[0]/(STOP_T - START_T)))
  periodic_log()
  periodic_save()
  nsteps = sess.run(global_step) - step
  log.log('Steps this session %d, calc time %.3fs (%.02f%% of optimizer)' % (
    nsteps, calc_time[0], 100.*calc_time[0]/(STOP_T - START_T)))
  log.log('Calc steps/sec %.3f, with overhead steps/sec %.3f' % (
    nsteps/calc_time[0], nsteps/(STOP_T - START_T)))
  END_T = time.time()
  log.log('Total time spent: %.3fs' % (END_T - INIT_T))
  for k, v in [
      ('before main', MAIN_T - INIT_T),
      ('setting up', START_T - MAIN_T),
      ('optimizing', STOP_T - START_T),
      ('finishing up', END_T - STOP_T),
      ]:
    log.log(' - time spent %s: %.3fs (%.02f%% of total)' % (
      k, v, 100.*v/(END_T - INIT_T)))
  log.close()

if __name__ == '__main__':
  main()

# vim:set ts=2 sw=2 sts=2 et:
