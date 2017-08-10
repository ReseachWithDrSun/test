"""
mainwindow.py
"""
import os
import re
import sys
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QMessageBox
from gallerywidget import GalleryWidget


class MainWindow(QtGui.QMainWindow):
    # custom signals must be class variables in PyQt
    sig_closing = QtCore.pyqtSignal()  # signal window close to shut down inspect dialog as well
    # signal gallery changes to update inspect dialog, args: record_id, tag_id
    # TODO: better way to pass current tag id
    sig_sample_updated = QtCore.pyqtSignal(int, int)

    def __init__(self, image_tagger):
        super(MainWindow, self).__init__()
        self.tagger = image_tagger
        self.current_tag = ''  # current tag selected in combo box

        uic.loadUi(os.path.join(self.tagger.dir_path, 'mainwindow.ui'), self)

        # dynamically add threshold control
        self.sliderThreshold = QtGui.QSlider(Qt.Horizontal, parent=self)
        self.sliderThreshold.setObjectName('sliderThreshold')
        self.sliderThreshold.focusOutEvent = lambda e: self.sliderThreshold.setVisible(False)
        self.sliderThreshold.setMinimum(0)
        self.sliderThreshold.setMaximum(100)
        self.sliderThreshold.setValue(50)
        self.sliderThreshold.setVisible(False)
        # and our gallery widget as well
        self.listPositive = GalleryWidget(self.centralWidget)
        self.positiveLayout.addWidget(self.listPositive)
        self.listNegative = GalleryWidget(self.centralWidget)
        self.negativeLayout.addWidget(self.listNegative)

        self.mainToolBar.setVisible(False)
        self.statusBar.setVisible(False)

        self.load_combo_tags()
        self.comboTag.setCurrentIndex(0)

        self.btnThreshold.clicked.connect(self.btnThreshold_clicked)
        self.btnLoad.clicked.connect(self.btnLoad_clicked)
        self.btnSave.clicked.connect(self.btnSave_clicked)
        self.btnPredict.clicked.connect(self.btnPredict_clicked)
        self.comboTag.currentIndexChanged.connect(self.comboTag_currentIndexChanged)
        self.sliderThreshold.valueChanged.connect(self.sliderThreshold_valueChanged)
        self.sliderZoom.valueChanged.connect(self.sliderZoom_valueChanged)
        self.listPositive.itemClicked.connect(self.listPositive_itemClicked)
        self.listNegative.itemClicked.connect(self.listNegative_itemClicked)

    """
    open the select dir dialog and reload images
    """
    def btnLoad_clicked(self):
        current_dir = QtGui.QFileDialog.getExistingDirectory(None, 'Select a folder', self.tagger.current_dir,
                                                             QtGui.QFileDialog.ShowDirsOnly)
        if current_dir != '':
            self.tagger.current_dir = current_dir
            self.tagger.load_dataset_from_dir()
            self.update_gallery()
            self.sig_sample_updated.emit(-1, self.comboTag.currentIndex())

    """
    load tags into combo box
    """
    def load_combo_tags(self):
        # clear current tags and images
        self.comboTag.clear()
        # clear image gallery

        self.tagger.load_tags() # reload all tags

        # add to combo box
        for tag in self.tagger.tags:
            if tag.parent != -1:
                self.comboTag.addItem('    ' + tag.name)
            else:
                self.comboTag.addItem(tag.name)
        self.comboTag_currentIndexChanged(0)    # manually trigger slot function for status update
        self.sig_sample_updated.emit(-1, self.comboTag.currentIndex())   # trigger inspect dialog update

    """
    save all tags to file
    """
    def btnSave_clicked(self):
        for record in self.tagger.records:
            record.tag.xml.write(record.tag_path)
        QMessageBox.information(self, 'ImageTagger', 'Save completed.')

    """
    run predict
    """
    def btnPredict_clicked(self):
        proceed = QMessageBox.warning(self, 'ImageTagger',
                                      'Running NN prediction will overwrite every tag to predict in all records, '
                                      'and revert manual corrections if you have made any.\n\nContinue?',
                                      QMessageBox.Ok | QMessageBox.Cancel)
        if proceed == QMessageBox.Ok:
            self.tagger.run_predict()
            self.update_gallery()
            self.sig_sample_updated.emit(-1, self.comboTag.currentIndex())

    """
    selected tag changed thru combo box; toggle threshold slider and update gallery

    Args:
        index: int, new combo index (tag id)
    """
    def comboTag_currentIndexChanged(self, index):
        ex_pos_ctrls = [self.labelPos1, self.labelPos2, self.labelPos3]
        ex_neg_ctrls = [self.labelNeg1, self.labelNeg2, self.labelNeg3]
        ex_default_path = os.path.join(self.tagger.dir_path, 'examples/default.jpg')
        ex_size = QtCore.QSize(128, 128)
        if index < 0:
            self.current_tag = ''
            # update the examples
            ex_default = QtGui.QPixmap.fromImageReader(QtGui.QImageReader(ex_default_path))
            for ctrl in ex_pos_ctrls:
                ctrl.setPixmap(ex_default.scaled(ex_size, Qt.KeepAspectRatioByExpanding))
            for ctrl in ex_neg_ctrls:
                ctrl.setPixmap(ex_default.scaled(ex_size, Qt.KeepAspectRatioByExpanding))
        else:
            self.current_tag = self.tagger.tags[self.comboTag.currentIndex()].name
            # update the examples
            for i, ctrl in enumerate(ex_pos_ctrls):
                ex_path = os.path.join(self.tagger.dir_path,
                                       'examples/' + re.sub(r'[ :]', '_', self.current_tag) + '_p' + str(i) + '.jpg')
                if not os.path.isfile(ex_path):
                    ex_path = ex_default_path
                ex = QtGui.QPixmap.fromImageReader(QtGui.QImageReader(ex_path))
                ctrl.setPixmap(ex.scaled(ex_size, Qt.KeepAspectRatioByExpanding))
            for i, ctrl in enumerate(ex_neg_ctrls):
                ex_path = os.path.join(self.tagger.dir_path,
                                       'examples/' + re.sub(r'[ :]', '_', self.current_tag) + '_n' + str(i) + '.jpg')
                if not os.path.isfile(ex_path):
                    ex_path = ex_default_path
                ex = QtGui.QPixmap.fromImageReader(QtGui.QImageReader(ex_path))
                ctrl.setPixmap(ex.scaled(ex_size, Qt.KeepAspectRatioByExpanding))
        self.btnThreshold.setEnabled(self.tagger.tags[self.comboTag.currentIndex()].to_predict)
        self.update_gallery()
        self.sig_sample_updated.emit(-1, self.comboTag.currentIndex())

    """
    display the threshold slider.
    """
    def btnThreshold_clicked(self):
        self.sliderThreshold.move(
            self.geometry().width() - self.sliderThreshold.geometry().width(),
            self.btnThreshold.geometry().top() + self.btnThreshold.geometry().height()
        )
        # read and set threshold
        threshold = [m[1] for m in self.tagger.predict_conf['output_map']
                     if m[0] == self.tagger.tags[self.comboTag.currentIndex()].name][0]
        self.sliderThreshold.setValue(int(threshold * 100))
        self.sliderThreshold.setVisible(True)
        self.sliderThreshold.setFocus()

    """
    save the new threshold.

    Args:
        value: new threshold
    """
    def sliderThreshold_valueChanged(self, value):
        for i in range(len(self.tagger.predict_conf['output_map'])):
            if self.tagger.predict_conf['output_map'][i][0] == self.tagger.tags[self.comboTag.currentIndex()].name:
                self.tagger.predict_conf['output_map'][i][1] = value / 100
                break
        self.tagger.save_config()
        self.tagger.do_thresholding()
        self.update_gallery()
        self.sig_sample_updated.emit(-1, self.comboTag.currentIndex())

    """
    change the display size of thumbnails.

    Args:
        value: new size
    """
    def sliderZoom_valueChanged(self, value):
        self.listPositive.setIconSize(QtCore.QSize(value, value))
        self.listNegative.setIconSize(QtCore.QSize(value, value))

    """
    reinsert image galleries with selected tag
    """
    def update_gallery(self):
        # clear current samples in both lists
        while self.listPositive.count() > 0:
            self.listPositive.takeItem(0)
        while self.listNegative.count() > 0:
            self.listNegative.takeItem(0)
        # place samples according to new tag
        for record in self.tagger.records:
            t = QtGui.QListWidgetItem(QtGui.QIcon(record.thumb_path), record.name)
            # just shrink the image names so that they don't take up too much space
            f = QtGui.QFont()
            f.setPointSize(1)
            t.setFont(f)
            if record.tag.contains(self.current_tag):
                self.listPositive.addItem(t)
            else:
                self.listNegative.addItem(t)

    """
    take the update signal from dialog inspect to (maybe) update galleries

    Args:
        tag_id: int
        record_id: int
    """
    def update_from_dialog_inspect(self, tag_id, record_id):
        # TODO: better ways to decide whether gallery update is necessary
        # only update the galleries when the tag updated matches the current tag in main window
        # or one is parent of another (which may trigger consistency check)
        if self.current_tag != self.tagger.tags[tag_id].name and \
            self.tagger.tags[self.comboTag.currentIndex()].parent != tag_id and \
                self.tagger.tags[tag_id].parent != self.comboTag.currentIndex():
            return
        try:
            # find the item, send it to another gallery and highlight
            if self.tagger.records[record_id].tag.contains(self.current_tag):
                item = self.listNegative.findItems(self.tagger.records[record_id].name, Qt.MatchExactly)[0]
                # a little detour because Qt has inconsistent params for adding/removing item...
                self.listNegative.setCurrentItem(item)
                it = self.listNegative.takeItem(self.listNegative.currentRow()) # item
                self.listPositive.addItem(it)
                self.listPositive.setCurrentItem(it)
                self.listNegative.setCurrentItem(None)
            else:
                item = self.listPositive.findItems(self.tagger.records[record_id].name, Qt.MatchExactly)[0]
                self.listPositive.setCurrentItem(item)
                it = self.listPositive.takeItem(self.listPositive.currentRow())
                self.listNegative.addItem(it)
                self.listNegative.setCurrentItem(it)
                self.listPositive.setCurrentItem(None)
        except IndexError:
            # if update is triggered by parent/sub tag changed and current tag is not affected, just ignore it
            print('No gallery update is needed.', file=sys.stderr)

    """
    handle clicks on samples

    Args:
        item: QListWidgetItem, item from itemClicked signal
    """
    def listPositive_itemClicked(self, item):
        index = [i for i, record in enumerate(self.tagger.records) if item.text() == record.name][0]

        if self.listPositive.mouse_button == Qt.LeftButton:
            # left click: cancel selection in another list
            self.listNegative.setCurrentItem(None)
        elif self.listPositive.mouse_button == Qt.RightButton:
            # right click: send it away
            self.tagger.records[index].tag.remove(self.current_tag)
            self.tagger.check_tag_consistency(self.tagger.records[index],
                                              update_major=self.tagger.tags[self.comboTag.currentIndex()].parent > -1)
            it = self.listPositive.takeItem(self.listPositive.currentRow())   # it's just *item*
            self.listNegative.addItem(it)
            # move selection
            self.listPositive.setCurrentItem(None)
            self.listNegative.setCurrentItem(it)

        self.sig_sample_updated.emit(index, self.comboTag.currentIndex()) # update inspect dialog

    def listNegative_itemClicked(self, item):
        index = [i for i, record in enumerate(self.tagger.records) if item.text() == record.name][0]

        if self.listNegative.mouse_button == Qt.LeftButton:
            # left click: cancel selection in another list
            self.listPositive.setCurrentItem(None)
        elif self.listNegative.mouse_button == Qt.RightButton:
            # right click: send it away
            self.tagger.records[index].tag.add(self.current_tag)
            self.tagger.check_tag_consistency(self.tagger.records[index],
                                              update_major=self.tagger.tags[self.comboTag.currentIndex()].parent > -1)
            it = self.listNegative.takeItem(self.listNegative.currentRow())   # it's just *item*
            self.listPositive.addItem(it)
            # move selection
            self.listNegative.setCurrentItem(None)
            self.listPositive.setCurrentItem(it)

        self.sig_sample_updated.emit(index, self.comboTag.currentIndex()) # update inspect dialog

    def closeEvent(self, event):
        self.sig_closing.emit()
