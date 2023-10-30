import argparse
import os
import traceback
import shutil
import logging
import yaml
import torch
import sys
import numpy as np

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


parser = argparse.ArgumentParser()


parser.add_argument('--dataset', type=str, default="Imagenette")
parser.add_argument('--attack_method', type=str, default="BadNet")
parser.add_argument( "--dataset_config", type=str,  default= "imagenet_256.yml", help="Path to the config file")
parser.add_argument( "--attack_schedule", type=str,  default= "BadNet", help="Path to the config file")
parser.add_argument('--datasets_root_dir', type=str, default="datasets")
parser.add_argument('--classes', type=int, default= 20)
parser.add_argument('--img_size', type=int, default=84)

parser.add_argument('--testwclean', type=bool, default=False)
parser.add_argument('--testwpoisoned', type=bool, default=False)
parser.add_argument('--testwpurified', type=bool, default=True)
parser.add_argument('-ptra','--purify_traindataset', action='store_true')
parser.add_argument('-pctes','--purify_clean_test', action='store_true')
parser.add_argument('-pptes','--purify_pois_test',  action='store_true')
parser.add_argument('-uptra','--use_purified_train', action='store_true')
parser.add_argument('-upctes','--use_purified_clean_test', action='store_true')
parser.add_argument('-upptes','--use_purified_pois_test',  action='store_true')
parser.add_argument('--mode', type=int, default=0)
parser.add_argument('--concat', type=bool, default=True)

parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
parser.add_argument("--deg", type=str, default= "sr_averagepooling", help="Liner_transformation types")
parser.add_argument("--path_y", type=str, default= "attack", help="Path of the test dataset.")
parser.add_argument("--exp", type=str, default="exp", help="Path for saved pre-trained diffusion model.")
parser.add_argument("--at_threshold", type=int, default=10, help="at_threshold" )    
parser.add_argument("--simplified", default="True", help="Use simplified DDNM, without SVD")    
parser.add_argument("--image_folder", type=str,default="images", help="The folder name of samples")
parser.add_argument("--image_test_folder", type=str,default="images", help="The folder name of samples")
parser.add_argument("--image_test_folder_pois", type=str,default="images", help="The folder name of samples")
parser.add_argument("--splited_image_folder", type=str,default="images", help="The folder name of samples")
parser.add_argument("--splited_test_image_folder", type=str,default="images", help="The folder name of samples")
parser.add_argument("--splited_test_image_folder_pois", type=str,default="images", help="The folder name of samples")
parser.add_argument("--apy_folder", type=str, default="./apy", help="The folder name of samples with linear transformation")
parser.add_argument("--deg_scale", type=float, default=2., help="transformation level, such as: the degree of blur")    
parser.add_argument("--ni", default=True, help="No interaction")
parser.add_argument("--deterministic", type=bool, default=True)
parser.add_argument('--timesteps', type=int, default=1000)
parser.add_argument('--sampling', type=int, default=20)

parser.add_argument("--pur_folder", type=str, default="./pur", help="The folder name of purified images samples")
parser.add_argument("--splited_pur_folder", type=str, default="./pur_splited", help="The folder name of splited image samples")
parser.add_argument('--dataset_seed', type=int, default=42)
parser.add_argument('--gpu', type=str, default="cuda:0")
parser.add_argument('--gpulist', nargs='+', type=int, default=[0,1,2])
parser.add_argument('--useAVGUP', type=bool, default=False)

base_args = parser.parse_args()

with open(os.path.join("configs", base_args.dataset_config), "r") as f:
    config = yaml.safe_load(f)
base_config = dict2namespace(config)

print(f"Using the {base_args.dataset}")
print(f"The at_threshold: {base_args.at_threshold}")


def parse_args_and_config(args, new_config):

        
    if args.use_purified_train == True:
        
            assert(args.use_purified_train == True)
            
            if args.use_purified_clean_test == True and args.use_purified_pois_test == True:
                
                
                args.mode = 1
                
            elif args.use_purified_clean_test == False and args.use_purified_pois_test == False:
                
                args.mode = 2
                
    elif args.use_purified_train == False:
            
            
            if args.use_purified_clean_test == True and args.use_purified_pois_test == True:
                
                args.mode = 3
               
            elif args.use_purified_clean_test == False and args.use_purified_pois_test == False:
                
                args.mode = 4
    '''
    The following is to create folder the train, val, val_pois, respectively.
    '''

    args.image_folder = os.path.join(
        args.pur_folder, f"Mode{args.mode}", f"{args.dataset}", f"{args.attack_method}", f"{args.deg}", f"{args.deg_scale}",f"{args.at_threshold}", "train"
    )
    args.splited_image_folder = os.path.join(
        args.splited_pur_folder, f"Mode{args.mode}", f"{args.dataset}", f"{args.attack_method}", f"{args.deg}", f"{args.deg_scale}",f"{args.at_threshold}", "train"
    )
    
    if not os.path.exists(args.splited_image_folder):
        os.makedirs(args.splited_image_folder)
    else:
        if args.ni:
            if args.purify_traindataset:
                shutil.rmtree(args.splited_image_folder)
                os.makedirs(args.splited_image_folder)
        
    #print(args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            if args.purify_traindataset:
                overwrite = True
                shutil.rmtree(args.image_folder)
                os.makedirs(args.image_folder)
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)")
            if response.upper() == "Y":
                overwrite = True
                
    args.test_image_folder = os.path.join(
        args.pur_folder, f"Mode{args.mode}", f"{args.dataset}", f"{args.attack_method}", f"{args.deg}", f"{args.deg_scale}", f"{args.at_threshold}","val" #follow the naming convention in the tinyimage net
    )
    args.splited_test_image_folder = os.path.join(
        args.splited_pur_folder, f"Mode{args.mode}", f"{args.dataset}", f"{args.attack_method}", f"{args.deg}", f"{args.deg_scale}", f"{args.at_threshold}", "val"
    )
    
    if not os.path.exists(args.splited_test_image_folder):
        os.makedirs(args.splited_test_image_folder)
    else:
        if args.ni:
            if args.purify_clean_test:
                shutil.rmtree(args.splited_test_image_folder)
                os.makedirs(args.splited_test_image_folder)
        
    if not os.path.exists(args.test_image_folder):
        os.makedirs(args.test_image_folder)
    else:
        if args.ni:
            if args.purify_clean_test:
                shutil.rmtree(args.test_image_folder)
                os.makedirs(args.test_image_folder)
        else:
            response = input(
                f"Image folder {args.test_image_folder} already exists. Overwrite? (Y/N)")
            if response.upper() == "Y":
                overwrite = True
                

    args.test_image_folder_pois = os.path.join(
        args.pur_folder, f"Mode{args.mode}", f"{args.dataset}", f"{args.attack_method}", f"{args.deg}", f"{args.deg_scale}", f"{args.at_threshold}", "val_pois" #follow the naming convention in the tinyimage net
    )
    args.splited_test_image_folder_pois = os.path.join(
        args.splited_pur_folder, f"Mode{args.mode}", f"{args.dataset}", f"{args.attack_method}", f"{args.deg}", f"{args.deg_scale}", f"{args.at_threshold}", "val_pois"
    )
    
    if not os.path.exists(args.splited_test_image_folder_pois):
        os.makedirs(args.splited_test_image_folder_pois)
    else:
        if args.ni:
            if args.purify_pois_test:
                shutil.rmtree(args.splited_test_image_folder_pois)
                os.makedirs(args.splited_test_image_folder_pois)
        
    #print(args.test_image_folder_pois)
    if not os.path.exists(args.test_image_folder_pois):
        os.makedirs(args.test_image_folder_pois)
    else:
        overwrite = False
        if args.ni:
            if args.purify_pois_test:
                overwrite = True
                shutil.rmtree(args.test_image_folder_pois)
                os.makedirs(args.test_image_folder_pois)
        else:
            response = input(
                f"Image folder {args.test_image_folder_pois} already exists. Overwrite? (Y/N)")
            if response.upper() == "Y":
                overwrite = True
                
    '''
    The following is the original image and augment images
    '''                  
                
    args.train_image_folder_apy = os.path.join(
        args.pur_folder, f"Mode{args.mode}", f"{args.dataset}", f"{args.attack_method}", f"{args.deg}", f"{args.deg_scale}", f"{args.at_threshold}", "train_apy" #follow the naming convention in the tinyimage net
    )
    args.train_image_folder_orig = os.path.join(
        args.pur_folder, f"Mode{args.mode}", f"{args.dataset}", f"{args.attack_method}", f"{args.deg}", f"{args.deg_scale}", f"{args.at_threshold}", "train_orig" #follow the naming convention in the tinyimage net
    )
    
    if not os.path.exists(args.train_image_folder_apy):
        os.makedirs(args.train_image_folder_apy)
    else:
        if args.ni:
            if args.purify_traindataset:
                shutil.rmtree(args.train_image_folder_apy)
                os.makedirs(args.train_image_folder_apy)
        
    #print(args.test_image_folder_pois)
    if not os.path.exists(args.train_image_folder_orig):
        os.makedirs(args.train_image_folder_orig)
    else:
        overwrite = False
        if args.ni:
            if args.purify_traindataset:
                shutil.rmtree(args.train_image_folder_orig)
                os.makedirs(args.train_image_folder_orig)
                
                
    args.test_image_folder_apy = os.path.join(
        args.pur_folder, f"Mode{args.mode}", f"{args.dataset}", f"{args.attack_method}", f"{args.deg}", f"{args.deg_scale}", f"{args.at_threshold}", "val_apy" 
    )
    args.test_image_folder_orig = os.path.join(
        args.pur_folder, f"Mode{args.mode}", f"{args.dataset}", f"{args.attack_method}", f"{args.deg}", f"{args.deg_scale}", f"{args.at_threshold}", "val_orig" 
    )
    
    if not os.path.exists(args.test_image_folder_apy):
        os.makedirs(args.test_image_folder_apy)
    else:
        if args.ni:
            if args.purify_clean_test:
                shutil.rmtree(args.test_image_folder_apy)
                os.makedirs(args.test_image_folder_apy)
        
    #print(args.test_image_folder_pois)
    if not os.path.exists(args.test_image_folder_orig):
        os.makedirs(args.test_image_folder_orig)
    else:
        if args.ni:
            if args.purify_clean_test:
                shutil.rmtree(args.test_image_folder_orig)
                os.makedirs(args.test_image_folder_orig)
                
                
    args.test_pois_image_folder_apy = os.path.join(
        args.pur_folder, f"Mode{args.mode}", f"{args.dataset}", f"{args.attack_method}", f"{args.deg}", f"{args.deg_scale}", f"{args.at_threshold}", "val_pois_apy" 
    )
    args.test_pois_image_folder_orig = os.path.join(
        args.pur_folder, f"Mode{args.mode}", f"{args.dataset}", f"{args.attack_method}", f"{args.deg}", f"{args.deg_scale}", f"{args.at_threshold}", "val_pois_orig" 
    )
    
    if not os.path.exists(args.test_pois_image_folder_apy):
        os.makedirs(args.test_pois_image_folder_apy)
    else:
        if args.ni:
            if args.purify_pois_test:
                shutil.rmtree(args.test_pois_image_folder_apy)
                os.makedirs(args.test_pois_image_folder_apy)
        
    #print(args.test_image_folder_pois)
    if not os.path.exists(args.test_pois_image_folder_orig):
        os.makedirs(args.test_pois_image_folder_orig)
    else:
        if args.ni:
            if args.purify_pois_test:
                shutil.rmtree(args.test_pois_image_folder_orig)
                os.makedirs(args.test_pois_image_folder_orig)

    device = torch.device(f"{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config





