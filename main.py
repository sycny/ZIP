
import traceback
import shutil
import logging
import yaml
import sys
import json

import torch
import numpy as np

# from runners.diffusion import Diffusion
from attack.backdoorattack import process_dataset, clean_ins, purified_ins
from preprocess.guided_diffusion.purify import  Purify, SplitDataset, nonSplitDataset, SplitCLeanDataset

from settings import base_args, base_config, parse_args_and_config

torch.set_printoptions(sci_mode=False)



def main():
    
    args, config = parse_args_and_config(base_args, base_config)
    
    config.diffusion.num_diffusion_timesteps = args.timesteps
    config.time_travel.T_sampling = args.sampling
    
    print(f"Mode is {args.mode}")
    
    '''
        There are four main modes; 
        (1) The model is trained with purified_train_dataset, and we purify both the two kinds of test datasets (clean/poisoned).
        (2) The model is trained with purified_train_dataset, we use the original two kinds of test datasets (clean/poisoned).
        (3) The model is trained with poisoned_train_dataset, and we purify both the two clean/poisoned test datasets to see if (1) maintain the CA, and (2) decrease the ASR.
        (4) The model is trained with poisoned_train_dataset, and we test with clean and poisoned iamges without purification.
    '''
    
    #read the original datasets, and attack the original datasets.
    
    clean_train, clean_test, poisoned_train, poisoned_test, backdoor_instance = process_dataset(args)
    
    #take the attacked datasets as input, return the purified datasets.
    
    '''
    In this part, we will first prepare the datasets.
    '''
    if args.purify_traindataset:
        
        train_runner = Purify(args, config, "train", poisoned_train)
        purified_train =  train_runner.pur()
              
    else:
        if args.use_purified_train:
            if args.concat:
                purified_train = SplitDataset(poisoned_train, args, "train")
            else:
                purified_train = nonSplitDataset(args, "train")

            
    if args.purify_clean_test:
        
        clean_test_runner  = Purify(args, config, "test", clean_test)
        purified_clean_test =  clean_test_runner.pur()

    else:
        if args.use_purified_clean_test:
            if args.concat:
                purified_clean_test = SplitCLeanDataset(clean_test, args, "test")
            else:
                purified_clean_test = nonSplitDataset(args, "test")    
                
    if args.purify_pois_test:
        
        pois_test_runner  = Purify(args, config, "test_pois", poisoned_test)
        purified_pois_test =  pois_test_runner.pur()

    else:
        if args.use_purified_pois_test:
            if args.concat:
                purified_pois_test = SplitDataset(poisoned_test, args, "test_pois")
            else:
                purified_pois_test = nonSplitDataset(args, "test_pois")     
             
    
    #Now we have the 1. original (train_dataset, test_dataset); 2. attacked (train_dataset, test_dataset); 3. purified (train_dataset, test_dataset)
    
    '''
    In this part, we will prepare the clean/attacked models.
    '''
    
    #let us first get the model using clean dataset as training datasets, for each datasets, we only need to run once. 
    if args.testwclean:
        clean_instance = clean_ins(args, clean_train, clean_test)
        print("====================================================================================")
        print("Train with clean, test with clean")
        with open("experiment_schedule.json", "r") as f:
            schedules = json.load(f)
            schedule = schedules['Clean']
            f.close()
            
        schedule['save_dir'] = args.dataset + '_' + "pretrain"
        clean_instance.train(schedule)
        print("====================================================================================")
        
        
        
    #let us then get the model using poisoned dataset as training datasets, for each datasets and attack kind, we only need to run once.
    if args.testwpoisoned:
        print("====================================================================================")
        print("Train with poisoned, test with clean and poisoned")
        with open("experiment_schedule.json", "r") as f:
            schedules = json.load(f)
            schedule = schedules[f'{args.attack_method}']
            f.close()
        
        
        schedule['save_dir'] = args.dataset + '_' + "pretrain"
        backdoor_instance.train(schedule)
        print("====================================================================================")
        
    
    #let us then get the purified CA and ASR, if args.use_purified_train is false, we will not re-train any model, and we will only inference model with different test datasets
    '''
    In this part, we will use the clean/attacked models to test our purified images.
    '''
    if args.testwpurified:
        '''
        There are four main modes; 
        (1) The model is trained with purified_train_dataset, and we purify both the two kinds of test datasets (clean/poisoned).
        (2) The model is trained with purified_train_dataset, we use the original two kinds of test datasets (clean/poisoned).
        (3) The model is trained with poisoned_train_dataset, and we purify both the two clean/poisoned test datasets to see if (1) maintain the CA, and (2) decrease the ASR.
        (4) The model is trained with poisoned_train_dataset, and we test with clean and poisoned iamges without purification.
        '''
        print("====================================================================================")
        if args.use_purified_train is True:
            
            if args.use_purified_clean_test is True and args.use_purified_pois_test is True:
                
                purified_instance = purified_ins(args, clean_train, purified_clean_test, purified_train, purified_pois_test) # this mode is not discussed in the ZIP
                print("purified_ins(args, clean_train, purified_clean_test, purified_train, purified_pois_test)") 
                     
            elif args.use_purified_clean_test is False and args.use_purified_pois_test is False:
                
                purified_instance = purified_ins(args, clean_train, clean_test, purified_train, poisoned_test)  # this mode is not discussed in the ZIP
                print("purified_ins(args, clean_train, clean_test, purified_train, poisoned_test)")
        
        elif args.use_purified_train is False: # using the poisoned_train_dataset to train the NN
            
            if args.use_purified_clean_test is True and args.use_purified_pois_test is True:
                
                purified_instance = purified_ins(args, clean_train, purified_clean_test, poisoned_train, purified_pois_test) # obtain the defense performance 
                print("purified_ins(args, clean_train, purified_clean_test, poisoned_train, purified_pois_test)")
                     
            elif args.use_purified_clean_test is False and args.use_purified_pois_test is False:
                
                purified_instance = purified_ins(args, clean_train, clean_test, poisoned_train, poisoned_test) # obtain the attack performance 
                print("purified_ins(args, clean_train, clean_test, poisoned_train, poisoned_test)")
    
        print("====================================================================================")
        
        with open("experiment_schedule.json", "r") as f:
            schedules = json.load(f)
            schedule = schedules['Purified']
            f.close()
            
        schedule['experiment_name'] = args.attack_method
        schedule['img_size'] = args.img_size
        schedule['save_dir'] = args.dataset + '_' + "pretrain"
        
        purified_instance.train(schedule)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
