'''
This is the implement of Purified images.
'''

import copy
import random

import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from attack.originalimagenet import Origdataset

from .base import *



class Purified(Base):

    def __init__(self, train_dataset, test_dataset, model, loss, schedule, seed, deterministic, poisoned_train_dataset,poisoned_test_dataset):

        super(Purified, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)

        self.poisoned_train_dataset = poisoned_train_dataset

        self.poisoned_test_dataset = poisoned_test_dataset
