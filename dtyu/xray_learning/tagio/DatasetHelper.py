import os
import shutil


class SyntheticDatasetHelper:
    def __init__(self, root):
        self.root = root

    def get_image_list(self):
        l = []
        for d in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, d)):
                for f in os.listdir(os.path.join(self.root, d)):
                    name, ext = os.path.splitext(f)
                    if ext == '.mat':
                        l.append(os.path.join(self.root, d, f))
        return l

    def get_tag_from_image(self, image_path):
        image_dir, image_name = os.path.split(image_path)
        name, ext = os.path.splitext(image_name)
        return os.path.join(image_dir, 'analysis/results/', name + '.xml')

    def split_expr_name(self, image_path):
        image_dir, image_name = os.path.split(image_path)
        name, ext = os.path.splitext(image_name)
        return image_dir.split('/')[-1], name

    def copy_to(self, image_path, dest_path):
        def try_mkdir(path):
            if not os.path.isdir(path):
                os.mkdir(path)

        try_mkdir(os.path.join(dest_path, 'analysis'))
        try_mkdir(os.path.join(dest_path, 'analysis/results'))
        try_mkdir(os.path.join(dest_path, 'analysis/thumbnails'))
        try_mkdir(os.path.join(dest_path, 'analysis/oned'))

        image_dir, image_name = os.path.split(image_path)
        name, ext = os.path.splitext(image_name)
        shutil.copyfile(image_path,
                        os.path.join(dest_path, name + '.mat'))
        shutil.copyfile(os.path.join(image_dir, 'analysis/results', name + '.xml'),
                        os.path.join(dest_path, 'analysis/results', name + '.xml'))
        shutil.copyfile(os.path.join(image_dir, 'analysis/thumbnails', name + '.jpg'),
                        os.path.join(dest_path, 'analysis/thumbnails', name + '.jpg'))
        shutil.copyfile(os.path.join(image_dir, 'analysis/oned', name + '.npy'),
                        os.path.join(dest_path, 'analysis/oned', name + '.npy'))


# for the "new" real dataset: 2016-3-13
class RealDatasetHelper:
    def __init__(self, root):
        self.root = root

    def get_image_list(self):
        l = []
        for f in os.listdir(self.root):
            name, ext = os.path.splitext(f)
            if ext == '.mat':
                l.append(os.path.join(self.root, f))
        return l

    def get_tag_from_image(self, image_path):
        image_dir, image_name = os.path.split(image_path)
        name, ext = os.path.splitext(image_name)
        # directory "relocation" for cropped real images
        # a new folder of generated raw data uses the original tags!
        return os.path.join(image_dir, '../mini_image/' + name + '.xml')

    def get_name(self, image_path):
        image_dir, image_name = os.path.split(image_path)
        name, ext = os.path.splitext(image_name)
        return name
