"""
main.py
"""
import sys
from PyQt4 import QtGui
from mainwindow import MainWindow
from dialoginspect import DialogInspect
from imagetagger import ImageTagger


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    image_tagger = ImageTagger()
    # bind the main image tagger object
    main_window = MainWindow(image_tagger)
    dialog_inspect = DialogInspect(image_tagger)
    # register signal-slots for updates across the two windows
    main_window.sig_closing.connect(dialog_inspect.quit)
    main_window.sig_sample_updated.connect(dialog_inspect.update_view)
    dialog_inspect.sig_tag_updated.connect(main_window.update_from_dialog_inspect)
    # place windows
    desk_geo = QtGui.QApplication.desktop().geometry()
    main_geo = main_window.geometry()
    dlg_geo = dialog_inspect.geometry()
    main_window.move((desk_geo.width() - main_geo.width() - dlg_geo.width()) / 2,
                     (desk_geo.height() - main_geo.height()) / 2)
    dialog_inspect.move((desk_geo.width() + main_geo.width() - dlg_geo.width()) / 2,
                        (desk_geo.height() - dlg_geo.height()) / 2)

    main_window.show()
    dialog_inspect.show()
    sys.exit(app.exec_())
