"""
Old viewer code using GTK3.
"""

def viewer(calc_fn=None, render_fn=None, fps=30, hang=True):
  """
  Opens a window which will display an ongoing process.

  calc_fn is called to do work incrementally. It should return at a rate of e.g.
  60fps. It should return True if it wants to be called again.

  render_fn is called when the window is ready for an update, it should return
  an image.

  If hang is True, viewer() returns after the window is closed (or the user hits
  Esc or q).

  Otherwise, viewer() stops when calc_fn first returns False.
  """
  # Do imports late so that users who don't use viewer() don't depend on these:
  import gi  # pip3 install pygobject
  gi.require_version('Gtk', '3.0')
  from gi.repository import GLib
  from gi.repository import Gtk
  from gi.repository import Gdk
  from gi.repository import GdkPixbuf

  im = Gtk.Image()
  window = Gtk.Window()
  last_time = [0]  # wrap in list because closures, zero so first update renders
  frame_secs = 1. / fps  # framerate cap
  res = [200,100]

  def idle_fn():
    try:
      cont = False
      if calc_fn is not None:
        cont = calc_fn()
        if not hang and not cont: Gtk.main_quit()
      if render_fn is not None:
        now = time.time()
        if now > last_time[0] + frame_secs or not cont:
          last_time[0] = now
          img = render_fn()
          assert img.dtype == np.uint8, img.dtype
          img = rgbify(img)
          h,w,_ = img.shape
          if [w,h] != res:
            # Resize and move.
            res[0] = w
            res[1] = h
            window.resize(w,h)
            window.move((Gdk.Screen.width() - w) // 2,
                        (Gdk.Screen.height() - h) // 2)
          have_alpha = False
          rowstride = w * 3
          pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
              GLib.Bytes.new_take(img.tostring()),
              GdkPixbuf.Colorspace.RGB, have_alpha, 8, w, h, rowstride)
          im.clear()
          im.set_from_pixbuf(pixbuf.copy())  # copy() reduces memory leaked!
          im.show()
      return cont  # true = keep running
    except:
      Gtk.main_quit()
      raise

  window.set_position(Gtk.WindowPosition.CENTER)  # center of screen
  window.set_decorated(True)  # False = no title bar or border
  window.set_resizable(True)
  # for tiling window managers:
  window.set_type_hint(Gdk.WindowTypeHint.UTILITY)
  window.connect('delete-event', Gtk.main_quit)  # on window close
  window.resize(*res)
  window.add(im)

  def on_key_press_event(widget, event):
    key = Gdk.keyval_name(event.keyval)
    if key in ('Escape', 'q'):
      Gtk.main_quit()
    else:
      print('unknown key pressed:', repr(key))
  window.connect('key_press_event', on_key_press_event)
  window.present()
  GLib.idle_add(idle_fn)
  Gtk.main()

# vim:set ts=2 sw=2 sts=2 et:
