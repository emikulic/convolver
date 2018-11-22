# Data and examples

* blur0.0.jpg - unblurred image
* blur1.2.jpg - Gaussian blur, radius 1.2, GIMP 2.10.8
* deblur.jpg - example output
* k1.2.npy - unblur kernel for radius 1.2 gaussian blur
* k1.2.png - visualization of kernel

The example output was generated with:

```shell
./conv.py blur1.2.jpg k1.2.npy -out deblur.png
```

The visualization was generated with:

```shell
./view.py k1.2.npy -out k1.2.png
```
