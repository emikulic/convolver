"""
Utility functions for kern.py.
"""
import os
try:
  os.environ['NO_TF']
except KeyError:
  import tensorflow as tf
import numpy as np
import time
import sys
from PIL import Image  # pip3 install pillow
from functools import reduce
import cairo  # pip3 install pycairo

# --- Core logic ---

def kernel_fn(kdir, step):
  return '%s/kernel-step%08d' % (kdir, step)

def save_kernel(kdir, step, kernel):
  sfn = kdir + '/step'
  wfn = kernel_fn(kdir, step)
  np.save(sfn, step)
  np.save(wfn, kernel)

def load_kernel(fn):
  """
  Returns step, kernel.
  """
  if os.path.isfile(fn):
    return 0, np.load(fn)
  step = np.load(fn + '/step.npy')
  k = np.load(kernel_fn(fn, step) + '.npy')
  return step, k

def load_image(fn, args, gray=True):
  """
  Loads an image from the given filename, applies gamma correction, and
  returns a numpy array of type float32.
  """
  img = np.asarray(Image.open(fn))
  img = to_float(img, args.gamma).astype(np.float32)
  if 'crop_x' in args or 'crop_y' in args or \
     'crop_w' in args or 'crop_h' in args:
    x,y = args.crop_x, args.crop_y
    w,h = args.crop_w, args.crop_h
    if h == 0: h = img.shape[0]
    if w == 0: w = img.shape[1]
    img = img[y:y+h, x:x+w, :]
  if gray:
    # Take the green channel.
    img = img[:, :, 1:2]
  h,w,c = img.shape
  return img.reshape([1,h,w,c])

def convolve(img, kern):
  return tf.nn.conv2d(img, kern, strides=[1,1,1,1],
      padding='SAME', data_format='NHWC')

def diff(actual, expected, border):
  """
  Returns a diff image.
  """
  return (rm_border(actual, border) -
          rm_border(expected, border))

def diff_cost(diff):
  """
  Returns a tf scalar that's the cost due to the difference between
  the two images.
  """
  return tf.reduce_mean(tf.square(diff))  # L2

def reg_cost(kern):
  """
  Returns a tf scalar that's the regularization cost of the kernel.
  """
  return tf.reduce_sum(tf.abs(kern))  # L1

def avg_px_err(dcost, gamma):
  return np.sqrt(np.power(dcost, 1./gamma)) * 255.

# --- Utilities ---

class Logger:
  """
  Logs to stdout as well as a logfile. Adds timestamps.
  """
  def __init__(self, logfn):
    self.f = open(logfn, 'a')
    self.last_time = time.time()
    pass

  def log(self, *args):
    now = time.time()
    dt = now - self.last_time
    self.last_time = now
    line = '%.3f (+%07.03f) %s %s\n' % (now, dt,
        time.strftime('%Y-%m-%d %H:%M:%S%z'),
        ' '.join(map(str, args)))
    sys.stdout.write(line)
    sys.stdout.flush()
    self.f.write(line)
    self.f.flush()

  def close(self):
    self.f.close()

def to_float(img, gamma=2.2):
  """
  Convert image from uint8 to floating point, does gamma correction.
  """
  out = img.astype(np.float) / 255.
  if gamma != 1.0:
    out = np.power(out, gamma)
  return out

def from_float(img, gamma=2.2):
  """
  Convert from floating point, doing gamma conversion and 0,255 saturation,
  into a byte array.
  """
  out = np.power(img.astype(np.float), 1.0 / gamma)
  out = (out * 255).clip(0, 255)
  # Rounding reduces quantization error (compared to just truncating).
  return np.round(out).astype(np.uint8)

def from_float_fast(img):
  """
  Convert from floating point, skipping gamma conversion, rounding, and
  clipping(!).
  """
  return (img * 255).astype(np.uint8)

def shape(t):
  """Returns shape of tensor as list of ints."""
  return [-1 if x is None else x for x in t.get_shape().as_list()]

def make_symmetric(a):
  # TODO: support making the kernel bigger.
  h,w,i,o = a.shape
  assert i == 1, i
  assert o == 1, o
  assert h == w, ['must be square', h, 'x', w]
  assert h % 2 == 1, ['size must be an odd number', h]
  sz = h // 2 + 1
  w = a[0:sz, 0:sz, :, :]
  w = tf.Variable(w)
  ur = tf.reverse(tf.slice(w, [0,0,0,0], [sz, sz-1, 1, 1]), [1])
  u = tf.concat([w, ur], 1)
  l = tf.reverse(tf.slice(u, [0,0,0,0], [sz-1, h, 1, 1]), [0])
  return tf.concat([u,l], 0)

def rm_border(t, border):
  """
  Removes `border` pixels from every edge, from NHWC tensor `t`.
  """
  n,h,w,c = shape(t)
  return t[:, border:h-border, border:w-border, :]

def lerp(a, b, x, u=0, v=1):
  "Linearly interpolate from a to b for x in [u,v]"
  if u == v:  # Degenerate case.
    return np.where(x < u, a, b)
  assert u < v
  x = np.clip(x, u, v)
  return a + (b-a) * (x-u) / (v-u)

assert lerp(1, 2, 100, 150, 150) == 1
assert lerp(1, 2, 200, 150, 150) == 2

def v(*x):
  """
  v() is for vector, returns a numpy array of floats.
  Example usage: v(0,1,0)
  """
  return np.asarray(x, dtype=float)

def bipolar(img):
  """
  Draw [-1,0] as soothing cerulean blue and [0,+1] as grayscale.
  """
  img = np.atleast_3d(img)
  neg = lerp(v(.02,.08,.12), v(.25,.5,1.), -img)
  pos = lerp(v(.02,.03,.04), v(1,1,1), img)
  out = np.zeros_like(pos)
  np.copyto(out, pos, 'same_kind', img > 0.)
  np.copyto(out, neg, 'same_kind', img < 0.)
  return out

def double(img):
  """Double width and height of all pixels."""
  shape = list(img.shape)
  shape[0] *= 2
  shape[1] *= 2
  out = np.zeros(shape, dtype=img.dtype)
  out[0::2, 0::2] = \
  out[0::2, 1::2] = \
  out[1::2, 0::2] = \
  out[1::2, 1::2] = img
  return out

def border(img, amount=1, value=0):
  """
  Adds requested amount of padding in each direction.
  """
  if len(img.shape) == 3:
    # Handle multiple channels.
    h,w,c = img.shape
    out = np.zeros((h+amount*2, w+amount*2, c), dtype=img.dtype)
    for i in range(c):
      out[:, :, i] = border(img[:, :, i], amount, value)
    return out
  h,w = img.shape
  out = np.zeros((h+amount*2, w+amount*2), dtype=img.dtype)
  out[0:amount, :] = value # top
  out[-amount:, :] = value # bottom
  out[amount:-amount, 0:amount] = value # left
  out[amount:-amount, -amount:] = value # right
  out[amount:amount+h, amount:amount+w] = img
  return out

def no_none(lst):
  return [i for i in lst if i is not None]

def hstack(lst, space=0):
  """
  Returns an image with the left and right pieces next to each other,
  aligned at the top edge, with an optional space in the middle.

  Like np.hstack() but heights can be mismatched, plus space.
  """
  def _hstack(left, right, space):
    assert left.dtype == right.dtype
    left = np.atleast_3d(left)
    right = np.atleast_3d(right)
    lh, lw, lc = left.shape
    rh, rw, rc = right.shape
    if lc != rc:
      left = rgbify(left)
      right = rgbify(right)
      lh, lw, lc = left.shape
      rh, rw, rc = right.shape
    assert lc == rc, ("channels mismatch", left.shape, right.shape)
    max_h = max(lh, rh)
    width = lw + space + rw
    out = np.zeros((max_h, width, lc), dtype=left.dtype)
    out[:lh, :lw, :lc] = left
    out[:rh, lw+space:lw+space+rw, :lc] = right
    return out
  return reduce(lambda l,r:_hstack(l,r,space), no_none(lst))

def vstack(lst, space=0):
  """
  Vertical version of hstack.
  """
  def _vstack(top, bottom, space):
    assert top.dtype == bottom.dtype
    top = np.atleast_3d(top)
    bottom = np.atleast_3d(bottom)
    th, tw, tc = top.shape
    bh, bw, bc = bottom.shape
    if tc != bc:
      top = rgbify(top)
      bottom = rgbify(bottom)
      th, tw, tc = top.shape
      bh, bw, bc = bottom.shape
    assert tc == bc, ('channels mismatch', top.shape, bottom.shape)
    max_w = max(tw, bw)
    height = th + space + bh
    out = np.zeros((height, max_w, tc), dtype=top.dtype)
    out[:th, :tw, :tc] = top
    out[th+space:th+space+bh, :bw, :tc] = bottom
    return out
  return reduce(lambda t,b:_vstack(t,b,space), no_none(lst))

def vis(fn, w=1, h=1, doubles=1, gamma=1.0):
  """
  Return visualization of 2D array returned by fn(i,j).
  Build a table of w x h tiles.
  Double the size of each tile some number of times.

  Example:
    # If you can't use vis_nhwc or vis_hwoi:
    l1_w = l1_w.reshape((in_n, l1_n, l1_n, l1_c))
    v = vis(lambda x,y: l1_w[x,:,:,y], in_n, l1_c, 0)
  """
  table = None
  white = 64
  for j in range(h):
    row = None
    for i in range(w):
      tile = bipolar(fn(i, j))
      if gamma != 1.0:
        tile = np.power(tile, 1.0 / gamma)
      tile = from_float_fast(tile)
      for _ in range(doubles):
        tile = double(tile)
      tile = border(tile, 1, white)
      row = hstack((row, tile), 2)
    table = vstack((table, row), 2)
  return table

def vis_nhwc(a, doubles=1, gamma=1.0):
  """
  Visualize a NHWC numpy array. (N is X, C is Y).
  """
  n,h,w,c = a.shape
  return vis(lambda x,y: a[x,:,:,y], n,c, doubles, gamma)

def vis_hwoi(t, doubles=1):
  """
  Visualize a HWOI kernel. (I is X, O is Y).
  """
  h,w,o,i = t.shape
  return vis(lambda x,y: t[:,:,y,x], i,o, doubles)

def make_label(text, size=10):
  """
  Returns a uint8 (h,w,1) of the text label rendered in white on a black
  background.
  """
  # measure text
  tmp = cairo.ImageSurface(cairo.FORMAT_A8, 1, 1)
  ctx1 = cairo.Context(tmp)
  ctx1.set_font_size(size)
  xb, yb, w, h = ctx1.text_extents(text)[:4]
  w = int(w)
  h = int(h)
  # make image
  img = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
  ctx = cairo.Context(img)
  ctx.set_font_size(size)
  # render text
  ctx.move_to(-xb, -yb)
  ctx.show_text(text)
  # convert to numpy array
  out = np.frombuffer(img.get_data(), np.uint8)
  out.shape = (img.get_height(), img.get_stride())
  return out[:h, :w].reshape((h,w,1))

def memoize(fn):
  """
  Use as decorator:

  @memoize
  def fac(x): ...
  """
  fn.cache = {}
  def memoized(*args):
    key = args
    try:
      return fn.cache[key]
    except KeyError:
      val = fn(*args)
      fn.cache[key] = val
      return val
  return memoized

cache_label = memoize(make_label)

def tf_init():
  """
  Force tf to initialize in a subdued tone.
  """
  sys.stderr.write('\033[30;1m')
  sys.stderr.flush()
  tf.constant(0)
  sys.stderr.write('\033[0m')
  sys.stderr.flush()

def save_image(fn, img):
  assert img.dtype == np.uint8
  h,w,c = img.shape
  if c == 1:
    im = Image.fromarray(img.reshape([h,w]), mode='L')
  else:
    im = Image.fromarray(img)
  # disable (png) compression to make this faster
  im.save(fn, optimize=False, compress_level=0)

def rgbify(i):
  """
  Convert grayscale to RGB by duplicating into 3 channels.
  This is a no-op on images that already have 3 channels.
  """
  i = np.atleast_3d(i)
  h,w,c = i.shape
  if c == 3:
    return i
  out = np.zeros((h,w,3), dtype=i.dtype)
  out[:,:,:] = i[:,:]
  return out

assert np.all(rgbify(np.array(
  [[1,2,3],
   [4,5,6]])) == [
     [[1,1,1],[2,2,2],[3,3,3]],
     [[4,4,4],[5,5,5],[6,6,6]]])

# vim:set ts=2 sw=2 sts=2 et:
