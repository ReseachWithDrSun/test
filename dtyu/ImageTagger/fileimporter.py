"""
fileimporter.py
An abstract File Importer class to support batch loading from a directory
"""
import os
import scipy.io
# import numpy as np
# from PIL import Image
from tag import tagdata, tagtype
dir_path = os.path.dirname(os.path.realpath(__file__))


class FileImporter:
    # initializer with a base directory
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_dir = data_dir  # path to image files (can be a subdir of the dataset)

    # signature method to check if dataset type matches
    @staticmethod
    def signature(data_dir):
        pass

    # load data from directory
    def load(self):
        pass

    @staticmethod
    def get_importer(data_dir):
        for importer in _supported:
            if importer.signature(data_dir):
                return importer(data_dir)
        return None


class FileImporterSynthetic(FileImporter):
    """
    File importer for Synthetic Dataset
    """
    @staticmethod
    def signature(data_dir):
        return os.path.isdir(os.path.join(data_dir, 'analysis'))

    def load(self):
        records = []

        class Record:
            pass

        for filename in os.listdir(self.image_dir):
            if os.path.isfile(os.path.join(self.image_dir, filename)):
                name, ext = os.path.splitext(filename)
                if ext == '.mat':
                    # data record found; fetch corresponding tag and thumbnail
                    record = Record()
                    record.name = name
                    record.data = scipy.io.loadmat(os.path.join(self.image_dir, filename))['detector_image']
                    record.tag_path = os.path.join(self.image_dir, 'analysis/results/' + name + '.xml')
                    record.thumb_path = os.path.join(self.image_dir, 'analysis/thumbnails/' + name + '.jpg')
                    if os.path.isfile(record.tag_path):
                        record.tag = tagdata(record.tag_path, tag_type=tagtype.Synthetic)
                    else:
                        # no tag file available; generate from template
                        record.tag = tagdata(os.path.join(dir_path, 'config/synthetic_template.xml'),
                                             tag_type=tagtype.Synthetic)
                    #record.thumb = np.array(Image.open(record.thumb_path))
                    records.append(record)

        return records, self.image_dir


class FileImporterReal(FileImporter):
    """
    File importer for Real Dataset
    """
    @staticmethod
    def signature(data_dir):
        return os.path.isdir(os.path.join(data_dir, 'mini_image'))

    def load(self):
        records = []

        class Record:
            pass

        self.image_dir = os.path.join(self.data_dir, 'mini_image')
        for filename in os.listdir(self.image_dir):
            if os.path.isfile(os.path.join(self.image_dir, filename)):
                name, ext = os.path.splitext(filename)
                if ext == '.mat':
                    # data record found; fetch corresponding tag and thumbnail
                    record = Record()
                    record.name = name
                    record.data = scipy.io.loadmat(os.path.join(self.image_dir, filename))['detector_image']
                    record.tag_path = os.path.join(self.image_dir, name + '.xml')
                    record.thumb_path = os.path.join(self.image_dir, name + '.jpg')
                    if os.path.isfile(record.tag_path):
                        record.tag = tagdata(record.tag_path, tag_type=tagtype.Synthetic)
                    else:
                        # no tag file available; generate from template
                        record.tag = tagdata(os.path.join(dir_path, 'config/synthetic_template.xml'),
                                             tag_type=tagtype.Synthetic)
                    #record.thumb = np.array(Image.open(record.thumb_path))
                    records.append(record)

        return records, self.image_dir


# TODO
'''
class FileImporterXMD(FileImporter):
    """
    File importer for XMD Dataset
    """
    @staticmethod
    def signature(data_dir):
        return os.path.isdir(os.path.join(data_dir, 'data')) and os.path.isdir(os.path.join(data_dir, 'tag'))

    def load(self):
        records = []

        class Record:
            pass

        self.image_dir = os.path.join(self.data_dir, 'mini_image')
        for filename in os.listdir(self.image_dir):
            if os.path.isfile(os.path.join(self.image_dir, filename)):
                name, ext = os.path.splitext(filename)
                if ext == '.mat':
                    # data record found; fetch corresponding tag and thumbnail
                    record = Record()
                    record.name = name
                    record.data = scipy.io.loadmat(os.path.join(self.image_dir, filename))['detector_image']
                    record.tag_path = os.path.join(self.image_dir, name + '.xml')
                    record.thumb_path = os.path.join(self.image_dir, name + '.jpg')
                    if os.path.isfile(record.tag_path):
                        record.tag = tagdata(record.tag_path, tag_type=tagtype.Synthetic)
                    else:
                        # no tag file available; generate from template
                        record.tag = tagdata(os.path.join(dir_path, 'config/synthetic_template.xml'),
                                             tag_type=tagtype.Synthetic)
                    #record.thumb = np.array(Image.open(record.thumb_path))
                    records.append(record)

        return records, self.image_dir
'''


_supported = {FileImporterSynthetic, FileImporterReal}
