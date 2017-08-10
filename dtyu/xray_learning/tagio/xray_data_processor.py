""" preprocess the raw data with tags and convert to tensorflow feed-ready format
new binary format: label + 1d + imagedata (tentatively all uint8)
"""

import numpy as np
import os
import scipy.io
import re
from PIL import Image
from six.moves import xrange
from tagio.tag import * #tagdata, tagtype


# load and parse all tag files in a directory
def process_xray_tags(tag_dir, image_dir='', prefix=None, tag_type=tagtype.Real):
    tags = []
    for root, dirs, files in os.walk(tag_dir):
        for name in files:
            filename, extname = os.path.splitext(name)
            if tag_type == tagtype.Real and extname == '.tag':
                tag = tagdata(os.path.join(root, name), tag_type=tagtype.Real)
                image_path = os.path.join(image_dir, tag.name)
                if os.path.isfile(image_path):    # verify image file
                    tag.ImagePath = image_path
                    tags.append(tag)
            elif tag_type == tagtype.Synthetic and extname == '.xml':
                tag = tagdata(os.path.join(root, name), prefix=prefix, tag_type=tagtype.Synthetic)
                image_path = os.path.join(image_dir, tag.name)
                image_path = image_path.replace('/analysis/results', '')
                image_path = image_path.replace('.xml', '.mat')
                #print(image_path)
                #raise Exception()
                if os.path.isfile(image_path):    # verify image file
                    tag.ImagePath = image_path
                    tags.append(tag)
    if tag_type == tagtype.Real:
        print(AllTags)
    return tags


def _filter_image(image, imax=2**16):
    # log transform and dtype conversion (open to placeholder style input maybe)
    # imax now a param to control range
    # probably use uint8 to save space
    image = 255 * np.log(1 + image) / np.log(imax)#np.log(image) / np.log(1.03)
    image = image.astype(np.uint8)
    return image


# preprocess tiff based real data
def _process_xray_images_tif(tags, image_dir, output_filename, size=None, offset=32):
    with open(output_filename, 'wb') as f:
        for tag in tags:
            image = Image.open(tag.ImagePath)
            if size is not None:
                image = image.resize(size)
            imarray = np.array(image)
            imarray = _filter_image(imarray)
            f.write(bytes([0] * offset))
            f.write(imarray.tostring())


# preprocess mat based synthetic data
def _process_xray_images_mat(tags, image_dir, output_filename, size=None, offset=32):
    with open(output_filename, 'wb') as f:
        for i, tag in enumerate(tags):
            if i % 100 == 0:
                print('%d / %d' % (i, len(tags)))
            image = scipy.io.loadmat(tag.ImagePath)['detector_image']
            image = _filter_image(image, 2**16)
            #if size is not None:    # to-do: resize will need to convert to Image to process!
            #    image = image.resize(size)
            # if images is None:
            #     images = np.expand_dims(image, axis=0)
            # else:
            #     images = np.insert(images, images.shape[0], image, axis=0)
            f.write(bytes([0] * offset))
            f.write(image.tostring())
    #_tags = [i for j, i in enumerate(_tags) if j not in _ind_noimage]    # remove tags w/o images


# preprocess tagged images and update valid tags
def process_xray_images(tags, image_dir, output_filename, size=None, tag_type=tagtype.Real, tag_only=False, offset=32):
    if tag_type == tagtype.Real:
        _process_xray_images_tif(tags, image_dir, output_filename, size=size, offset=offset)
    elif tag_type == tagtype.Synthetic:
        _process_xray_images_mat(tags, image_dir, output_filename, size=size, offset=offset)
    else:
        raise ValueError('Unrecognized tag type')


# update the 1D curves in a binary save
def update_oned(tags, filename, record_length, label_length):
    with open(filename, 'r+b') as f:
        for i, tag in enumerate(tags):
            tag_dir = os.path.dirname(tag.path)
            tag_dir = tag_dir.replace('analysis/results', '')
            tag_name, _ = os.path.splitext(os.path.basename(tag.path))
            oned_path = os.path.join(tag_dir, 'analysis/oned/' + tag_name + '.npy')
            oned = np.load(oned_path).astype(np.float32)
            oned[np.isinf(oned) | np.isnan(oned)] = 0.
            f.seek(i * record_length + label_length)
            f.write(oned.tostring())


# generate the image labels with the feature specified
def generate_xray_labels(tags, label_selector, attrib=None, label_len=32):
    labels = np.zeros([len(tags), label_len], dtype=np.uint8)
    for i, tag in enumerate(tags):
        labels[i,:] = label_selector(tag, attrib)
    return labels


def RegexFeatureSelector(feature_list, regex_list):
    assert len(regex_list) <= 32, 'RegEx list max size exceeded'
    label = np.zeros([1, 32], dtype=np.uint8)
    for i in xrange(len(regex_list)):
        if regex_list[i] != '':
            label[0, i] = any(re.search(regex_list[i], x) for x in feature_list)
    return label


def MainImageFeatureSelector(tag, attrib=None):
    patterns = [r'low-q',
        r'high-q',
        r'Halo',
        r'Higher orders',
        r'Ring',
        '',
        '',
        '',
        '',
        '',
        r'Symmetry ring',
        r'Circular beamstop',
        '',
        r'Linear beamstop',
        r'Beam off image']
    return RegexFeatureSelector(tag.tags, patterns)


def SimulatedFeatureSelector(tag, attrib=None):
    patterns = [r'low-q',
        r'high-q',
        r'halo',
        r'higher orders',
        r'main.ring',
        r'BCC',
        r'FCC',
        r'hexagonal',
        r'lamellar',
        #r'lattice SC',
        r'symmetry halo',
        r'symmetry ring',
        r'circular beamstop',
        r'wedge beamstop',
        r'linear beamstop',
        r'beam off image']
    return RegexFeatureSelector(tag.SimulatedFeatures, patterns)


def SimulatedFeatureSelectorSymmetry(tag, attrib=None):
    #  0      1       2     3              4      5              6              7      8      9
    # [low-q, high-q, halo, higher orders, rings, symmetry halo, symmetry ring, 2-sym, 4-sym, 6-sym]
    patterns = [r'low-q',
        r'high-q',
        r'halo',
        r'higher orders',
        r'main.ring',
        r'symmetry halo',
        r'symmetry ring']
    label = np.zeros([1, 32], dtype=np.uint8)
    for i, p in enumerate(patterns):
        label[0, i] = any(re.search(p, x) for x in tag.SimulatedFeatures)
    symmetry_tags = [x for x in tag.SimulatedFeatures if x.find('symmetry') > -1]
    for t in symmetry_tags:
        symmetry_order = int(t[t.find(': ') + 2])
        if symmetry_order == 2:
            label[0, 7] = 1
        elif symmetry_order == 4:
            label[0, 8] = 1
        elif symmetry_order == 6:
            label[0, 9] = 1
    return label


# update the labels in a binary save
# record_length: the length of a whole record to skip ahead
def update_labels(filename, labels, record_length):
    with open(filename, 'r+b') as f:
        for i in xrange(labels.shape[0]):
            f.seek(i * record_length)
            f.write(labels[i,:].tostring())
