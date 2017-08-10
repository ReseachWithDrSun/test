"""
imagetagger.py
The processor class of the app
"""
import os
import sys
import yaml
from fileimporter import FileImporter
from tag import MismatchedAttribute
# can't stuff this in the class for reusing because of scope issues
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # load prediction module config
    with open(os.path.join(dir_path, 'config/predict.yml'), 'r') as f:
        sys.path.append(yaml.load(f)['predict_module_path'])
    from nn import nn_eval2 as nn_eval
except yaml.YAMLError as err:
    print(err)
except ImportError:
    print('Failed to load NN prediction module; predict is unavailable', file=sys.stderr)


class ImageTagger:
    dir_path = os.path.dirname(os.path.realpath(__file__))  # dir of this file (the program)

    """
    TagName class is used to organize tag hierarchy.
    """
    class TagName:
        def __init__(self, name):
            self.name = name
            self.parent = -1  # list index of parent tag
            self.subtags = []  # list of subtags
            self.to_predict = False  # tag is to be predicted by NN

    def __init__(self):
        self.predict_conf = {}  # NN predict config
        self.tags = []  # all tags used in the app
        self.records = []  # all data (tags, image, thumb)
        self.predict = None  # NN prediction
        self.current_dir = '.'  # current dir from load
        self.image_dir = ''  # dir of image data (can be a subdir of current_dir)
        self.load_config()

    """
    load nn prediction config
    """
    def load_config(self):
        try:
            # load prediction module config
            self.predict_conf = yaml.load(open(os.path.join(self.dir_path, 'config/predict.yml'), 'r'))
        except yaml.YAMLError as err:
            print(err)

    """
    save nn prediction config
    """
    def save_config(self):
        with open(os.path.join(self.dir_path, 'config/predict.yml'), 'w') as f:
            f.write(yaml.dump(self.predict_conf))

    """
    load all tag names from config file
    """
    def load_tags(self):
        self.tags.clear()
        tags_to_predict = [m[0] for m in self.predict_conf['output_map']]
        with open(os.path.join(self.dir_path, 'config/tagnames'), 'r') as f:
            tagnames = f.read().splitlines()
        tid = 0
        last_parent = 0
        for tagname in tagnames:
            if '\t' in tagname:
                # a subtag
                tag = self.TagName(tagname.replace('\t', ''))
                tag.parent = last_parent
                tag.to_predict = tag.name in tags_to_predict
                self.tags[last_parent].subtags.append(tid)
                self.tags.append(tag)
            else:
                tag = self.TagName(tagname)
                tag.to_predict = tag.name in tags_to_predict
                last_parent = tid
                self.tags.append(tag)
            tid += 1

    """
    load all data records (tag, image, thumb) from directory
    """
    def load_dataset_from_dir(self):
        self.predict = None
        self.records.clear()
        self.records, self.image_dir = FileImporter.get_importer(self.current_dir).load()
        # consistency check
        for record in self.records:
            self.check_tag_consistency(record, update_major=True)

    """
    call nn prediction model
    """
    def run_predict(self):
        names = [s.name for s in self.records]
        # change working dir for the TF backend to find the correct relative dir
        wd = os.getcwd()
        os.chdir(self.predict_conf['predict_module_path'])
        self.predict, _ = nn_eval.evaluate(self.image_dir, names)
        os.chdir(wd)
        self.do_thresholding()

    """
    thresholding true/false output from predict probs
    """
    def do_thresholding(self):
        if self.predict is None:
            return
        for i in range(self.predict.shape[0]):
            for j in range(self.predict.shape[1]):
                try:
                    if self.predict[i][j] > self.predict_conf['output_map'][j][1]:
                        self.records[i].tag.add(self.predict_conf['output_map'][j][0])
                    else:
                        self.records[i].tag.remove(self.predict_conf['output_map'][j][0])
                except MismatchedAttribute:
                    pass    # tag value already exists
            # subtags are updated here because major tags are predicted
            self.check_tag_consistency(self.records[i], update_major=False)

    """
    a(n incomplete) consistency check of major/minor tag hierarchy.
    fix two cases: (1) set major tag according to status of all of its subtags
                   (2) erase minor tags if major tag is cancelled/predicted negative
    there is no natural way to "enforce" consistency if false positive is predicted or user decides to toggle on a new
    major tag.

    Args:
        record: data record, with the tag record to check
        update_major: bool, perform (1) if true, (2) o/w
    """
    def check_tag_consistency(self, record, update_major=True):
        tag_major = [tag for tag in self.tags if len(tag.subtags) > 0]  # extract all major tags with subtags
        for tag in tag_major:
            if update_major:
                try:
                    if any([record.tag.contains(self.tags[t].name) for t in tag.subtags]):
                        record.tag.add(tag.name)
                    else:
                        record.tag.remove(tag.name)
                except MismatchedAttribute:
                    pass
            else:
                if not record.tag.contains(tag.name):
                    for sub_id in tag.subtags:
                        try:
                            record.tag.remove(self.tags[sub_id].name)
                        except MismatchedAttribute:
                            pass
