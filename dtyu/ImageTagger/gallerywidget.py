"""
gallerywidget.py
Inherited widget of QListWidget to track mouse keys with click events to
provide support for different functions for left/right clicks on items
"""
from PyQt4.QtCore import Qt, QSize
from PyQt4.QtGui import QListWidget, QListView


class GalleryWidget(QListWidget):
    mouse_button = 0

    def __init__(self, parent=None):
        super(GalleryWidget, self).__init__(parent)
        # preset properties
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMovement(QListView.Static)
        self.setResizeMode(QListView.Adjust)
        self.setViewMode(QListView.IconMode)
        self.setIconSize(QSize(64, 64))
        self.setSortingEnabled(True)

    def mousePressEvent(self, event):
        self.mouse_button = event.button()
        super(GalleryWidget, self).mousePressEvent(event)   # send back to base class handler
