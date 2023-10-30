import torchvision
import attack.BackdoorBox as bb
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip,  Resize
from torchvision.transforms import Compose, ToTensor,  RandomHorizontalFlip, ColorJitter, RandomAffine
from .originalimagenet import Origdataset


def read_image(img_path, type=None):
    img = cv2.imread(img_path)
    if type is None:
        return img
    elif isinstance(type, str) and type.upper() == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(type, str) and type.upper() == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError

def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid

def process_dataset(args):

    trainset = None
    testset = None

    default_img_height = args.img_size # setting img height
    default_img_width = args.img_size # setting img width

    if args.dataset == "Cifar10":
        dataset = torchvision.datasets.CIFAR10
        transform_train = Compose([
            Resize((default_img_height, default_img_width)),
            ToTensor()
        ])
        trainset = dataset(args.datasets_root_dir, train=True, transform=transform_train, download=True)

        transform_test = Compose([
            Resize((default_img_height, default_img_width)),
            ToTensor()
        ])
        testset = dataset(args.datasets_root_dir, train=False, transform=transform_test, download=True)

        
    elif args.dataset == "GTSRB":
        
        
        transform_train = Compose([
            Resize((default_img_height, default_img_width)),
            ToTensor()
        ])

        trainset = Origdataset(args, args.datasets_root_dir,  split="train", transform=transform_train)

        transform_test = Compose([
            Resize((default_img_height, default_img_width)),
            ToTensor()
        ])

        testset = Origdataset(args, args.datasets_root_dir,  split="val",  transform=transform_test)
        
    else:
        
        transform_train = Compose([
            Resize((default_img_height, default_img_width)),
            ToTensor()
        ])

        trainset = Origdataset(args, args.datasets_root_dir,  split="train",  transform=transform_train)

        transform_test = Compose([
            Resize((default_img_height, default_img_width)),
            ToTensor()
        ])

        testset = Origdataset(args, args.datasets_root_dir,  split="val",  transform=transform_test)

    
    if args.attack_method == "BadNet":
        
        pattern = torch.zeros((3, default_img_height, default_img_width), dtype=torch.uint8)
        weight = torch.zeros((3, default_img_height, default_img_width), dtype=torch.float32) 
        
        if default_img_height <= 128:
            pattern[:, -6:-3, -6:-3] = torch.randn((3,3,3)) * 255
            weight[:, -6:-3, -6:-3] = 1.0
        else:
            print("Using the pattern size: 9*9")
            pattern[:, -12:-3, -12:-3] = torch.randn((3,9,9)) * 255
            weight[:, -12:-3, -12:-3] = 1.0

        backdoor_instance = bb.core.BadNets(
            train_dataset=trainset,
            test_dataset=testset,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            y_target=1,
            poisoned_rate=0.05,
            pattern=pattern,
            weight=weight,
            seed=args.seed,
            deterministic=args.deterministic,
            poisoned_transform_train_index=1,
            poisoned_transform_test_index=1
        )
        
    elif args.attack_method == "PhysicalBA":
        
        pattern = torch.zeros((3, default_img_height, default_img_width), dtype=torch.uint8)
        weight = torch.zeros((3, default_img_height, default_img_width), dtype=torch.float32) 
        
        if default_img_height <= 128:
            pattern[:, -6:-3, -6:-3] = torch.randn((3,3,3)) * 255
            weight[:, -6:-3, -6:-3] = 1.0
        else:
            print("Using the pattern size: 9*9")
            pattern[:, -12:-3, -12:-3] = torch.randn((3,9,9)) * 255
            weight[:, -12:-3, -12:-3] = 1.0

        backdoor_instance = bb.core.PhysicalBA(
            train_dataset=trainset,
            test_dataset=testset,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            y_target=1,
            poisoned_rate=0.05,
            pattern=pattern,
            weight=weight,
            seed=args.seed,
            deterministic=args.deterministic,
            poisoned_transform_train_index=1,
            poisoned_transform_test_index=1,
            physical_transformations = Compose([
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.2,contrast=0.2), 
            RandomAffine(degrees=10,translate=(0.1, 0.1), scale=(0.8, 0.9))])
        )

    elif args.attack_method == "Blended":
        pattern = torch.zeros((1, default_img_height, default_img_width), dtype=torch.uint8)
        pattern[0, :, :] = torch.randint(0, 255, size=(default_img_height, default_img_width))
        weight = torch.zeros((1, default_img_height, default_img_width), dtype=torch.float32)
        weight[0, :, :] = 0.2

        backdoor_instance = bb.core.Blended(
            train_dataset=trainset,
            test_dataset=testset,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            y_target=1,
            poisoned_rate=0.05,
            pattern=pattern,
            weight=weight,
            seed=args.seed,
            deterministic=args.deterministic,
            poisoned_transform_train_index=1,
            poisoned_transform_test_index=1
        )
    
    elif args.attack_method == "WaNet":

        identity_grid, noise_grid = gen_grid(default_img_height, 256)
        backdoor_instance = bb.core.WaNet(
            train_dataset=trainset,
            test_dataset=testset,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            y_target=0, 
            poisoned_rate=0.1,
            identity_grid=identity_grid,
            noise_grid=noise_grid,
            noise=False,
            seed=args.seed,
            deterministic=args.deterministic,
            poisoned_transform_train_index=1,
            poisoned_transform_test_index=1
        )

    else:
        raise NotImplementedError

    poisoned_train_dataset, poisoned_test_dataset = backdoor_instance.poisoned_train_dataset, backdoor_instance.poisoned_test_dataset

    return trainset, testset, poisoned_train_dataset, poisoned_test_dataset, backdoor_instance


def clean_ins(args, clean_train, clean_test):
    
        clean_instance = bb.core.Clean(
            train_dataset = clean_train,
            test_dataset = clean_test,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            schedule=None,
            seed=args.dataset_seed,
            deterministic=True)
        
        return clean_instance
    
def purified_ins(args, clean_train, clean_test, purified_train, purified_test):
    
        purified_instance = bb.core.Purified(
            train_dataset =clean_train,
            test_dataset= clean_test,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            schedule=None,
            seed=args.dataset_seed,
            deterministic=True,
            poisoned_train_dataset = purified_train,
            poisoned_test_dataset = purified_test)
        
        return  purified_instance
    
    