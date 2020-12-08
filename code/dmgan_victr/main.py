from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import warnings
warnings.filterwarnings('ignore')

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_DMGAN.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--NET_G', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    
    
    #VICTR
    parser.add_argument('--use_sg', action='store_true', default=False)

    
    
    args = parser.parse_args()
    return args




def gen_example(dataset, args, output_dir): #wordtoix
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    #filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    print("Reading filenames...")
    filepath = '%s/files.txt' % (cfg.DATA_DIR)

    data_dic = {}
    index_list=[]
    with open(filepath, "r") as f:
        #filenames = f.read().decode('utf8').split('\n')
        fnames = f.read().split('\n')
        for name in fnames:
            #print("#########")
            image_id=name.strip('\r')
            #print(image_id)
            if len(image_id)<1:
                continue
            image_index=dataset.filenames.index(image_id)
            index_list.append(image_index)
    sampler_analysis=SubsetRandomSampler(index_list)
    dataloader_analysis=torch.utils.data.DataLoader(dataset, batch_size=10, sampler=sampler_analysis)
    algo = trainer(output_dir, dataloader_analysis, dataset.n_words, dataset.ixtoword, \
        [dataset.index2objvec, \
        dataset.index2vec_inside, dataset.index2vec_outside, dataset.index2vec_left, \
        dataset.index2vec_right, dataset.index2vec_above, dataset.index2vec_below], \
        args.use_sg, dataset)
    algo.analysis(dataset, len(index_list), output_dir)



if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.NET_G != '':
        cfg.TRAIN.NET_G = args.NET_G

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    if not cfg.B_VALIDATION:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    torch.cuda.set_device(cfg.GPU_ID)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print("Seed: %d" % (args.manualSeed))

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform, use_sg=args.use_sg)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    print("define the traininer...")
    #algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword, dataset)
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword, \
        [dataset.index2objvec, \
        dataset.index2vec_inside, dataset.index2vec_outside, dataset.index2vec_left, \
        dataset.index2vec_right, dataset.index2vec_above, dataset.index2vec_below], \
        args.use_sg, dataset)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        '''generate images from pre-extracted embeddings'''
        if cfg.B_VALIDATION:
            algo.sampling(split_dir)  # generate images for the whole valid dataset
        else:
            gen_example(dataset, args, output_dir)  # generate images for customized captions
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
