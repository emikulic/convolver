#!/usr/bin/env python3
"""
Tests for tensorflow.nn.conv2d() behaviour with different sized kernels.
"""
import tensorflow as tf
import numpy as np
import unittest

def equal(a, b, threshold = 0.001):
  n,h,w,c = a.shape
  a = a.reshape((h,w))
  b = b.reshape((h,w))
  out = np.all(abs(a - b) < threshold)
  if not out:
    print('MISMATCH')
    print('got:')
    print(a)
    print('expected:')
    print(b)
  return out

class ConvTestCase(unittest.TestCase):
  def test_conv(self):
    in1 = np.array([
        [.0,.0,.0,.0,.0,.0],  # 0
        [.0,.0,.0,.0,.0,.0],  # 1
        [.0,1.,.0,2.,.0,.0],  # 2
        [.0,.0,.0,.0,.0,.0],  # 3
        [.0,.0,.0,1.,.0,.0],  # 4
        [.0,.0,.0,.0,.0,.0],  # 5
        [.0,.0,.0,.0,.0,.0],  # 6
    ], dtype=np.float32).reshape((1,7,6,1))  # NHWC

    # Odd-sized kernel.
    k1 = np.array([
        [.3,.4,.0],
        [.2,.0,.0],
        [.1,.0,.0],
    ], dtype=np.float32).reshape((3,3,1,1))  # HWOI

    out1a = np.array([
        [.0,.0,.0,.0,.0,.0],  # 0
        [.0,.0,.1,.0,.2,.0],  # 1
        [.0,.0,.2,.0,.4,.0],  # 2
        [.0,.4,.3,.8,.7,.0],  # 3
        [.0,.0,.0,.0,.2,.0],  # 4
        [.0,.0,.0,.4,.3,.0],  # 5
        [.0,.0,.0,.0,.0,.0],  # 6
    ], dtype=np.float32).reshape((1,7,6,1))

    # Same vs valid.
    conv1a = tf.nn.conv2d(in1, k1, strides = [1,1,1,1],
        padding='SAME', data_format='NHWC').numpy()

    assert equal(conv1a, out1a)

    out1b = np.array([
        [.0,.1,.0,.2],  # 1
        [.0,.2,.0,.4],  # 2
        [.4,.3,.8,.7],  # 3
        [.0,.0,.0,.2],  # 4
        [.0,.0,.4,.3],  # 5
    ], dtype=np.float32).reshape((1,5,4,1))

    # Valid vs same.
    conv1b = tf.nn.conv2d(in1, k1, strides = [1,1,1,1],
        padding='VALID', data_format='NHWC').numpy()

    assert equal(conv1b, out1b)

    # Even-sized kernel.
    k2 = np.array([
        [.1,.2],
        [.3,.4],
    ], dtype=np.float32).reshape((2,2,1,1))  # HWOI
    
    out2a = np.array([
        [.0,.0,.0,.0,.0,.0],  # 0
        [.4,.3,.8,.6,.0,.0],  # 1
        [.2,.1,.4,.2,.0,.0],  # 2
        [.0,.0,.4,.3,.0,.0],  # 3
        [.0,.0,.2,.1,.0,.0],  # 4
        [.0,.0,.0,.0,.0,.0],  # 5
        [.0,.0,.0,.0,.0,.0],  # 6
    ], dtype=np.float32).reshape((1,7,6,1))
    
    conv2a = tf.nn.conv2d(in1, k2, strides = [1,1,1,1],
        padding='SAME', data_format='NHWC').numpy()
    
    assert equal(conv2a, out2a)

    out2b = np.array([
        [.0,.0,.0,.0,.0],  # 0
        [.4,.3,.8,.6,.0],  # 1
        [.2,.1,.4,.2,.0],  # 2
        [.0,.0,.4,.3,.0],  # 3
        [.0,.0,.2,.1,.0],  # 4
        [.0,.0,.0,.0,.0],  # 5
    ], dtype=np.float32).reshape((1,6,5,1))
    
    conv2b = tf.nn.conv2d(in1, k2, strides = [1,1,1,1],
        padding='VALID', data_format='NHWC').numpy()

    assert equal(conv2b, out2b)

if __name__ == '__main__':
  unittest.main()

# vim:set ts=2 sw=2 sts=2 et:
