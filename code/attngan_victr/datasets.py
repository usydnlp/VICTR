from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import json
from miscc.utils import unicode_convert


def prepare_data(data, use_sg=False):

    if use_sg:
        imgs, captions, captions_lens, class_ids, keys, sg5_obj, sg5_att, sg5_rel, sg5_obj_count, \
                   sg5_obj_inside,sg5_rel_inside, \
                   sg5_obj_outside,sg5_rel_outside, \
                   sg5_obj_left,sg5_rel_left, \
                   sg5_obj_right,sg5_rel_right, \
                   sg5_obj_above,sg5_rel_above, \
                   sg5_obj_below,sg5_rel_below = data

    else:
        imgs, captions, captions_lens, class_ids, keys = data


    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    if use_sg:
        sg5_obj_count=sg5_obj_count[sorted_cap_indices]

        sg5_obj=sg5_obj[sorted_cap_indices]
        sg5_rel=sg5_rel[sorted_cap_indices]
        sg5_att=sg5_att[sorted_cap_indices]

        sg5_obj_inside=sg5_obj_inside[sorted_cap_indices]
        sg5_rel_inside=sg5_rel_inside[sorted_cap_indices]

        sg5_obj_outside=sg5_obj_outside[sorted_cap_indices]
        sg5_rel_outside=sg5_rel_outside[sorted_cap_indices]

        sg5_obj_left=sg5_obj_left[sorted_cap_indices]
        sg5_rel_left=sg5_rel_left[sorted_cap_indices]

        sg5_obj_right=sg5_obj_right[sorted_cap_indices]
        sg5_rel_right=sg5_rel_right[sorted_cap_indices]

        sg5_obj_above=sg5_obj_above[sorted_cap_indices]
        sg5_rel_above=sg5_rel_above[sorted_cap_indices]

        sg5_obj_below=sg5_obj_below[sorted_cap_indices]
        sg5_rel_below=sg5_rel_below[sorted_cap_indices]


        if cfg.CUDA:
            sg5_obj = Variable(sg5_obj).cuda()
            sg5_rel = Variable(sg5_rel).cuda()
            sg5_att = Variable(sg5_att).cuda()

            sg5_obj_inside = Variable(sg5_obj_inside).cuda()
            sg5_rel_inside = Variable(sg5_rel_inside).cuda()

            sg5_obj_outside = Variable(sg5_obj_outside).cuda()
            sg5_rel_outside = Variable(sg5_rel_outside).cuda()

            sg5_obj_left = Variable(sg5_obj_left).cuda()
            sg5_rel_left = Variable(sg5_rel_left).cuda()

            sg5_obj_right = Variable(sg5_obj_right).cuda()
            sg5_rel_right = Variable(sg5_rel_right).cuda()

            sg5_obj_above = Variable(sg5_obj_above).cuda()
            sg5_rel_above = Variable(sg5_rel_above).cuda()

            sg5_obj_below = Variable(sg5_obj_below).cuda()
            sg5_rel_below = Variable(sg5_rel_below).cuda()
        else:
            sg5_obj = Variable(sg5_obj)
            sg5_rel = Variable(sg5_rel)  

            sg5_obj_inside = Variable(sg5_obj_inside)
            sg5_rel_inside = Variable(sg5_rel_inside) 

            sg5_obj_outside = Variable(sg5_obj_outside)
            sg5_rel_outside = Variable(sg5_rel_outside) 

            sg5_obj_left = Variable(sg5_obj_left)
            sg5_rel_left = Variable(sg5_rel_left) 

            sg5_obj_right = Variable(sg5_obj_right)
            sg5_rel_right = Variable(sg5_rel_right) 

            sg5_obj_above = Variable(sg5_obj_above)
            sg5_rel_above = Variable(sg5_rel_above) 

            sg5_obj_below = Variable(sg5_obj_below)
            sg5_rel_below = Variable(sg5_rel_below) 
        

        return [real_imgs, captions, sorted_cap_lens, \
        class_ids, keys, sg5_obj, sg5_att, sg5_rel, sg5_obj_count, \
        sg5_obj_inside,sg5_rel_inside, \
                   sg5_obj_outside,sg5_rel_outside, \
                   sg5_obj_left,sg5_rel_left, \
                   sg5_obj_right,sg5_rel_right, \
                   sg5_obj_above,sg5_rel_above, \
                   sg5_obj_below,sg5_rel_below] 


    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys]


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None, use_sg=True):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        self.split=split
        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

        
        self.use_sg=use_sg 

        #[VICTR]:
        if self.use_sg: 
            
            
            #[VICTR] Basic graph
            self.index2objvec, self.sg2index = self.build_sg_vocab()
            self.index2obj=dict([val,key] for key,val in self.sg2index.items())
            self.index2rel=dict([val,key] for key,val in self.sg2index.items())
            self.index2att=dict([val,key] for key,val in self.sg2index.items())
            
            self.sg_object, self.sg_attribute, self.sg_relationship, self.obj_count = self.load_sg_index(split,'basic_sg')
            del self.sg2index
            
            
            #[VICTR] Positional graph
            self.index2vec_inside, self.name2index_inside, self.index2vec_outside, self.name2index_outside, \
            self.index2vec_left, self.name2index_left, self.index2vec_right, self.name2index_right, \
            self.index2vec_above, self.name2index_above, self.index2vec_below, self.name2index_below = self.build_pos_sg(self.data_dir)
            self.sg_object_inside, self.sg_relationship_inside, self.obj_count = self.load_sg_index(split, 'pos_inside')
            del self.name2index_inside
            self.sg_object_outside, self.sg_relationship_outside, self.obj_count = self.load_sg_index(split, 'pos_outside')
            del self.name2index_outside
            self.sg_object_left, self.sg_relationship_left, self.obj_count = self.load_sg_index(split, 'pos_left')
            del self.name2index_left
            self.sg_object_right, self.sg_relationship_right, self.obj_count = self.load_sg_index(split, 'pos_right')
            del self.name2index_right
            self.sg_object_above, self.sg_relationship_above, self.obj_count = self.load_sg_index(split, 'pos_above')
            del self.name2index_above
            self.sg_object_below, self.sg_relationship_below, self.obj_count = self.load_sg_index(split, 'pos_below')
            del self.name2index_below


    #VICTR_added
    def build_sg_vocab(self):

        GCN_file = os.path.join(self.data_dir, 'COCO_basic_graph.json')

        if not os.path.isfile(GCN_file):
            print('[VICTR_ERROR] No GCN embedding file found in %s ' % GCN_file)
        else:
            print('[VICTR] Loading Basic Graph embedding vocab from %s ...' % GCN_file)
            with open(GCN_file, 'r') as f:
                GCN_embed=json.load(f)

            padding=np.zeros((200,))
            index2vec = []
            index2vec.append(padding)
            sg2index= {}
            count=1
            for word in GCN_embed:
                    sg2index[word]=count
                    index2vec.append(GCN_embed[word])
                    count+=1  
        return np.array(index2vec).astype(np.float32), sg2index


    def build_pos_sg(self, data_dir):
        GCN_inside_file = os.path.join(data_dir, 'COCO_position_inside.json')
        GCN_outside_file = os.path.join(data_dir, 'COCO_position_outside.json')
        GCN_left_file = os.path.join(data_dir, 'COCO_position_left.json')
        GCN_right_file = os.path.join(data_dir, 'COCO_position_right.json')
        GCN_above_file = os.path.join(data_dir, 'COCO_position_above.json')
        GCN_below_file = os.path.join(data_dir, 'COCO_position_below.json')

        if not os.path.isfile(GCN_inside_file):
            print('[VICTR_ERROR]: No gcn embedding file found in %s ' % GCN_inside_file)
        if not os.path.isfile(GCN_outside_file):
            print('[VICTR_ERROR]: No gcn embedding file found in %s ' % GCN_outside_file)
        elif not os.path.isfile(GCN_left_file):
            print('[VICTR_ERROR]: No gcn embedding file found in %s ' % GCN_left_file)
        elif not os.path.isfile(GCN_right_file):
            print('[VICTR_ERROR]: No gcn embedding file found in %s ' % GCN_right_file)
        elif not os.path.isfile(GCN_above_file):
            print('[VICTR_ERROR]: No gcn embedding file found in %s ' % GCN_above_file)
        elif not os.path.isfile(GCN_below_file):
            print('[VICTR_ERROR]: No gcn embedding file found in %s ' % GCN_below_file)
        else:
            print('[VICTR] Loading Positional Graph embedding vocab')
            index2vec_inside, obj2index_inside=self.get_vocab(GCN_inside_file)
            index2vec_outside, obj2index_outside=self.get_vocab(GCN_outside_file)
            index2vec_left, obj2index_left=self.get_vocab(GCN_left_file)
            index2vec_right, obj2index_right=self.get_vocab(GCN_right_file)
            index2vec_above, obj2index_above=self.get_vocab(GCN_above_file)
            index2vec_below, obj2index_below=self.get_vocab(GCN_below_file)

        return np.array(index2vec_inside).astype(np.float32), obj2index_inside, np.array(index2vec_outside).astype(np.float32), obj2index_outside, \
               np.array(index2vec_left).astype(np.float32), obj2index_left, np.array(index2vec_right).astype(np.float32), obj2index_right, \
               np.array(index2vec_above).astype(np.float32), obj2index_above, np.array(index2vec_below).astype(np.float32), obj2index_below

    def get_vocab(self, file):
        with open(file, 'r') as f:
            embeddings=json.load(f)

        #can change to zeros_alike
        padding=np.zeros((50,))
        index2vec = []
        index2vec.append(padding)
        obj2index= {}
        count=1
        for word in embeddings:
                obj2index[word]=count
                index2vec.append(embeddings[word])
                count+=1  
        return index2vec, obj2index


    def load_sg_index(self, split, sg_embedding):
        data_dir = self.data_dir
        sg_path = os.path.join(data_dir, split, cfg.SG.SCENE_GRAPH_FILE)


        if sg_embedding=='basic_sg':
            sg_filepath = os.path.join(data_dir, split, 'sg_basic.pickle')
     
            obj2index, rel2index, att2index = self.sg2index, self.sg2index, self.sg2index
            return self.process_sg(sg_path, sg_filepath, obj2index, rel2index, att2index)
        elif sg_embedding=='pos_inside':
            sg_filepath = os.path.join(data_dir, split, 'sg_inside.pickle')
        
            obj2index, rel2index = self.name2index_inside, self.name2index_inside
        elif sg_embedding=='pos_outside':
            sg_filepath = os.path.join(data_dir, split, 'sg_outside.pickle')
         
            obj2index, rel2index = self.name2index_outside, self.name2index_outside
        elif sg_embedding=='pos_left':
            sg_filepath = os.path.join(data_dir, split, 'sg_left.pickle')
      
            obj2index, rel2index = self.name2index_left, self.name2index_left
        elif sg_embedding=='pos_right':
            sg_filepath = os.path.join(data_dir, split, 'sg_right.pickle')
       
            obj2index, rel2index = self.name2index_right, self.name2index_right
        elif sg_embedding=='pos_above':
            sg_filepath = os.path.join(data_dir, split, 'sg_above.pickle')
       
            obj2index, rel2index = self.name2index_above, self.name2index_above
        elif sg_embedding=='pos_below':
            sg_filepath = os.path.join(data_dir, split, 'sg_below.pickle')
       
            obj2index, rel2index = self.name2index_below, self.name2index_below
        else:
            sg_filepath = os.path.join(data_dir, split, 'sg_index.pickle')   
            obj2index, rel2index, att2index = self.obj2index, self.rel2index, self.att2index
            return self.process_sg(sg_path, sg_filepath, obj2index, rel2index, att2index)

        return self.process_sg(sg_path, sg_filepath, obj2index, rel2index)


    def process_sg(self, sg_path, sg_filepath, obj2index, rel2index, att2index=None):
        if not os.path.isfile(sg_filepath):
            with open(sg_path, 'r') as  f:
                sg=json.load(f)
            sg = unicode_convert(sg)

            print("[VICTR] Indexing scene graph for ", len(sg), " images")

            new_sg_obj=[]
            new_sg_att=[]
            new_sg_rel=[]
            new_sg_obj_count=[]
            for img in sg:
                sg_obj_per_image=[]
                sg_att_per_image=[]
                sg_rel_per_image=[]
                sg_obj_count_per_image=[]
                sgg5_object=[]
                sgg5_att=[]
                sgg5_rel=[]
                sgg5_obj_count=[]

                for a_sg in img:
                    mask_sg=[1]*cfg.SG.MAX_OBJECTS  #
                    obj_list=[0]*cfg.SG.MAX_OBJECTS #
                    rel_pad=[0]*cfg.SG.MAX_RELATIONSHIPS
                    rel_list=[rel_pad]*cfg.SG.MAX_OBJECTS
                    att_pad=[0]*cfg.SG.MAX_ATTRIBUTES
                    att_list=[att_pad]*cfg.SG.MAX_OBJECTS

                    if len(a_sg['objects'])>0:                     
                        if len(a_sg['objects'])<cfg.SG.MAX_OBJECTS:
                            mask_sg[:len(a_sg['objects'])]=[0]*len(a_sg['objects'])
                        else:
                            mask_sg[:]=[0]*cfg.SG.MAX_OBJECTS

                        get_rel=len(a_sg['relationships'])>0
                        get_att=len(a_sg['attributes'])>0
                        
                        #processing objects                        
                        for i,obj in enumerate(a_sg['objects']):
                            if i>cfg.SG.MAX_OBJECTS-1:
                                break
                            try:
                                obj_list[i]=obj2index[obj]
                            except KeyError:
                                obj_list[i]=0


                            #processing relationships for this object
                            if get_rel:
                                rel_pad=[0]*cfg.SG.MAX_RELATIONSHIPS
                                rel_ctr=[]
                                for rel in a_sg['relationships']:
                                    if rel[0]==i or rel[2]==i:
                                        try:
                                            rel_ctr.append(rel2index[rel[1]])
                                        except KeyError:
                                            rel_ctr.append(0)
                                if len(rel_ctr)<cfg.SG.MAX_RELATIONSHIPS:
                                    rel_pad[:len(rel_ctr)]=rel_ctr
                                else:
                                    rel_pad=rel_ctr[:cfg.SG.MAX_RELATIONSHIPS]
                                rel_list[i]=rel_pad

                            #processing attribute for this object
                            if get_att and att2index is not None:
                                att_pad=[0]*cfg.SG.MAX_ATTRIBUTES
                                att_ctr=[]
                                for att in a_sg['attributes']:
                                    if att[0]==i:
                                        try:
                                            att_ctr.append(att2index[att[2]])
                                        except KeyError:
                                            att_ctr.append(0)
                                if len(att_ctr)<cfg.SG.MAX_ATTRIBUTES:
                                    att_pad[:len(att_ctr)]=att_ctr
                                else:
                                    att_pad=att_ctr[:cfg.SG.MAX_ATTRIBUTES]
                                att_list[i]=att_pad


                        
                    assert len(mask_sg)==cfg.SG.MAX_OBJECTS
                    assert len(obj_list)==cfg.SG.MAX_OBJECTS
                    assert len(rel_list)==cfg.SG.MAX_OBJECTS
                    sgg5_obj_count.append(mask_sg)
                    sgg5_object.append(obj_list)
                    sgg5_rel.append(rel_list)
                    if att2index is not None:
                        sgg5_att.append(att_list)
                        assert len(att_list)==cfg.SG.MAX_OBJECTS

                sg_obj_per_image.append(sgg5_object)
                sg_rel_per_image.append(sgg5_rel)
                sg_obj_count_per_image.append(sgg5_obj_count)
                new_sg_obj.append(sg_obj_per_image)
                new_sg_rel.append(sg_rel_per_image)
                new_sg_obj_count.append(sg_obj_count_per_image)

                if att2index is not None:
                    sg_att_per_image.append(sgg5_att)
                    new_sg_att.append(sg_att_per_image)


            

            
            
            if att2index is not None:
                with open(sg_filepath, 'wb') as f:
                    pickle.dump([new_sg_obj, new_sg_att,
                                 new_sg_rel, new_sg_obj_count], f, protocol=2)
                    print('Save to: ', sg_filepath)

            else:
                with open(sg_filepath, 'wb') as f:
                    pickle.dump([new_sg_obj,
                                 new_sg_rel, new_sg_obj_count], f, protocol=2)
                    print('Save to: ', sg_filepath)

        else:
            
            print('[VICTR] Loading indexed scene graph from: ', sg_filepath)

            if att2index is not None:
                with open(sg_filepath, 'rb') as f:
                    x = pickle.load(f)
                    new_sg_obj, new_sg_att = x[0], x[1]
                    new_sg_rel, new_sg_obj_count = x[2], x[3]
                    del x
            else:
                with open(sg_filepath, 'rb') as f:
                    x = pickle.load(f)
                    new_sg_obj, new_sg_rel, new_sg_obj_count =x[0], x[1], x[2]
                    del x

            print("[VICTR] Loaded indexed scene graph for ", len(new_sg_obj), " images")

            

        if att2index is None:
            return new_sg_obj, new_sg_rel, new_sg_obj_count
        else:
            return new_sg_obj, new_sg_att, new_sg_rel, new_sg_obj_count



    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in xrange(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    #VICTR_added
    def get_sg(self, sent_ix):


        all_no_obj=False
        obj_to_check=self.obj_count[sent_ix][0] #5,5


        zero_obj=[]
        num_obj=[]
        for i,sgobjt in enumerate(obj_to_check):
            obj_nb=np.asarray(sgobjt).sum()
            num_obj.append(cfg.SG.MAX_OBJECTS-obj_nb)
            if obj_nb == cfg.SG.MAX_OBJECTS:
                zero_obj.append(i)

        if len(zero_obj)==5:
            all_no_obj=True

        if all_no_obj:
            selector = 0
        else:
            #while True:     
            #    selector=random.randint(0, self.embeddings_num)
            #    if selector not in zero_obj: 
            #        break
            selector=sorted(range(len(num_obj)), key=lambda k: num_obj[k], reverse=True)[0]


        sg5_obj = np.asarray(self.sg_object[sent_ix][0][selector]).astype('int64')
        sg5_att = np.asarray(self.sg_attribute[sent_ix][0][selector]).astype('int64')
        sg5_rel = np.asarray(self.sg_relationship[sent_ix][0][selector]).astype('int64')
        sg5_obj_count=np.asarray(self.obj_count[sent_ix][0][selector]).astype('int64') 


        sg5_obj_inside = np.asarray(self.sg_object_inside[sent_ix][0][selector]).astype('int64')
        sg5_rel_inside = np.asarray(self.sg_relationship_inside[sent_ix][0][selector]).astype('int64')

        sg5_obj_outside = np.asarray(self.sg_object_outside[sent_ix][0][selector]).astype('int64')
        sg5_rel_outside = np.asarray(self.sg_relationship_outside[sent_ix][0][selector]).astype('int64')

        sg5_obj_left = np.asarray(self.sg_object_left[sent_ix][0][selector]).astype('int64')
        sg5_rel_left = np.asarray(self.sg_relationship_left[sent_ix][0][selector]).astype('int64')

        sg5_obj_right = np.asarray(self.sg_object_right[sent_ix][0][selector]).astype('int64')
        sg5_rel_right = np.asarray(self.sg_relationship_right[sent_ix][0][selector]).astype('int64')

        sg5_obj_above = np.asarray(self.sg_object_above[sent_ix][0][selector]).astype('int64')
        sg5_rel_above = np.asarray(self.sg_relationship_above[sent_ix][0][selector]).astype('int64')

        sg5_obj_below = np.asarray(self.sg_object_below[sent_ix][0][selector]).astype('int64')
        sg5_rel_below = np.asarray(self.sg_relationship_below[sent_ix][0][selector]).astype('int64')

        return sg5_obj,sg5_att,sg5_rel,sg5_obj_count, \
               sg5_obj_inside,sg5_rel_inside, \
               sg5_obj_outside,sg5_rel_outside, \
               sg5_obj_left,sg5_rel_left, \
               sg5_obj_right,sg5_rel_right, \
               sg5_obj_above,sg5_rel_above, \
               sg5_obj_below,sg5_rel_below, selector




    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)


        if self.use_sg:
            
            sg5_obj,sg5_att,sg5_rel,sg5_obj_count, \
            sg5_obj_inside,sg5_rel_inside, \
            sg5_obj_outside,sg5_rel_outside, \
            sg5_obj_left,sg5_rel_left, \
            sg5_obj_right,sg5_rel_right, \
            sg5_obj_above,sg5_rel_above, \
            sg5_obj_below,sg5_rel_below,selector = self.get_sg(index)


            new_sent_ix = index * self.embeddings_num + selector
            caps, cap_len = self.get_caption(new_sent_ix)

            #get captions
            cp_id=list(caps[:cap_len])
            original_cap=" ".join([self.ixtoword[capidx[0]] for capidx in cp_id])
            selectedCapimg = os.path.join(self.data_dir, self.split, 'selectedCapimg.txt')
            selectedCap = os.path.join(self.data_dir, self.split, 'selectedCap.txt')
            with open(selectedCapimg, 'a') as f:
                f.write(key+": "+ original_cap+"\n\n")

            with open(selectedCap, 'a') as f:
                f.write(original_cap+"\n")

            return imgs, caps, cap_len, cls_id, key, sg5_obj, sg5_att, sg5_rel, sg5_obj_count, \
                   sg5_obj_inside,sg5_rel_inside, \
                   sg5_obj_outside,sg5_rel_outside, \
                   sg5_obj_left,sg5_rel_left, \
                   sg5_obj_right,sg5_rel_right, \
                   sg5_obj_above,sg5_rel_above, \
                   sg5_obj_below,sg5_rel_below

        else:
            sent_ix = random.randint(0, self.embeddings_num)
            new_sent_ix = index * self.embeddings_num + sent_ix
            caps, cap_len = self.get_caption(new_sent_ix)
            return imgs, caps, cap_len, cls_id, key



    def get_mis_caption(self, cls_id):
        mis_match_captions_t = []
        mis_match_captions = torch.zeros(99, cfg.TEXT.WORDS_NUM)
        mis_match_captions_len = torch.zeros(99)
        i = 0
        while len(mis_match_captions_t) < 99:
            idx = random.randint(0, self.number_example)
            if cls_id == self.class_id[idx]:
                continue
            sent_ix = random.randint(0, self.embeddings_num)
            new_sent_ix = idx * self.embeddings_num + sent_ix
            caps_t, cap_len_t = self.get_caption(new_sent_ix)
            mis_match_captions_t.append(torch.from_numpy(caps_t).squeeze())
            mis_match_captions_len[i] = cap_len_t
            i = i +1
        sorted_cap_lens, sorted_cap_indices = torch.sort(mis_match_captions_len, 0, True)

        for i in range(99):
            mis_match_captions[i,:] = mis_match_captions_t[sorted_cap_indices[i]]
        return mis_match_captions.type(torch.LongTensor).cuda(), sorted_cap_lens.type(torch.LongTensor).cuda()


    def __len__(self):
        return len(self.filenames)
