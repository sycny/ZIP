import os
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import verify_str_arg

class Origdataset(ImageFolder):
    """Dataset for Imagenette"""
    
    splits = ('train', 'val')

    def __init__(self, args, root,  split='train', download = False, **kwargs):
        self.args = args
        self.data_root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", self.splits)
        self.base_folder = args.dataset

        if not self._check_exists():
            raise RuntimeError('Dataset not found. See readme')

        super().__init__(self.split_folder, **kwargs)

    @property
    def dataset_folder(self):
        return os.path.join(self.data_root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.split)

    def _check_exists(self):
        return os.path.exists(self.split_folder)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

  
    


