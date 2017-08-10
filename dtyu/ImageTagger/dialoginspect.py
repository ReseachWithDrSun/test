"""
dialoginspect.py
"""
import os
import re
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import Qt
from PIL import Image, ImageQt
import numpy as np
from tag import MismatchedAttribute


class DialogInspect(QtGui.QDialog):
    # custom signals must be class variables in PyQt
    sig_tag_updated = QtCore.pyqtSignal(int, int)   # signal tag changes to update main window, args: tag_id, record_id

    def __init__(self, image_tagger):
        super(DialogInspect, self).__init__()
        self.tagger = image_tagger
        self.current_id = -1  # the current record in the inspect window
        self.lock_attribute = False  # disable itemChanged event when listTags is being updated in code

        uic.loadUi(os.path.join(self.tagger.dir_path, 'dialoginspect.ui'), self)

        # load processing line from config file
        try:
            self.textCmd.setText(open(os.path.join(self.tagger.dir_path, 'config/proccmd'), 'r').read())
        except FileNotFoundError:
            self.textCmd.setText('1 / np.log(2**16) * np.log(1 + image)')   # use default

        self.listTags.itemChanged.connect(self.listTags_itemChanged)

        self.quiting = False

    def closeEvent(self, event):
        # disable quitting unless main window is closed
        if not self.quiting:
            QtGui.QMessageBox.information(self, 'ImageTagger',
                                          'Close the main window to exit the app.')
            event.ignore()

    def quit(self):
        self.quiting = True
        self.close()

    """
    update image view and tags

    Args:
        record_id: int, id of record to display
        tag_id: int, current tag index
    """
    def update_view(self, record_id=None, tag_id=-1):
        if record_id is not None:
            self.current_id = record_id
        if self.current_id == -1:
            # deactivate controls
            self.lock_attribute = True
            self.listTags.clear()
            self.setWindowTitle('Inspect')
            # if tag_id > -1:
            #     # current tag available; fetch an example
            #     current_tag = self.tagger.tags[tag_id].name
            #     example_path = os.path.join(self.tagger.dir_path,
            #                                 'examples/' + re.sub(r'[ :]', '_', current_tag) + '.jpg')
            #     if os.path.isfile(example_path):
            #         image = QtGui.QPixmap.fromImageReader(QtGui.QImageReader(example_path))
            #         self.imageDisplay.setPixmap(image)
            #         self.imageDisplay.resize(self.imageDisplay.pixmap().size())
            #     else:
            #         self.imageDisplay.setText('No sample available')
        else:
            # update image
            image = self.tagger.records[self.current_id].data
            image = eval(self.textCmd.toPlainText())    # perform filter command
            image = (image / np.amax(image) * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image = QtGui.QPixmap.fromImage(ImageQt.ImageQt(image))
            self.imageDisplay.setPixmap(image)
            self.imageDisplay.resize(self.imageDisplay.pixmap().size())

            # update tag list
            self.lock_attribute = True  # disable itemChanged response before finishing update
            self.listTags.clear()
            for tag in self.tagger.tags:
                name = tag.name
                if tag.parent != -1:
                    name = '    ' + name    # indent for subtag
                t = QtGui.QListWidgetItem(name, self.listTags)
                t.setFlags(t.flags() | Qt.ItemIsUserCheckable)
                if self.tagger.records[self.current_id].tag.contains(tag.name):
                    t.setCheckState(Qt.Checked)
                else:
                    t.setCheckState(Qt.Unchecked)
            self.lock_attribute = False
            self.setWindowTitle('Inspect: ' + self.tagger.records[self.current_id].name)

    """
    update tags when check boxes are clicked

    Args:
        item: QListWidgetItem, item from itemChanged signal
    """
    def listTags_itemChanged(self, item):
        if self.lock_attribute:
            return
        name = item.text().lstrip()
        try:
            if item.checkState() == Qt.Checked:
                self.tagger.records[self.current_id].tag.add(name)
            else:
                self.tagger.records[self.current_id].tag.remove(name)
            self.tagger.check_tag_consistency(self.tagger.records[self.current_id],
                                              update_major=self.tagger.tags[self.listTags.row(item)].parent > -1)
            self.update_view()
            tag_id = [i for i, tag in enumerate(self.tagger.tags) if tag.name == name][0]
            self.sig_tag_updated.emit(tag_id, self.current_id) # signal the item to highlight
        except MismatchedAttribute:
            pass
