from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER, SG_Attention

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np
import sys
import json

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

# ################# Text to image task############################ #
class condGANTrainer(object):
    #def __init__(self, output_dir, data_loader, n_words, ixtoword, dataset):
    def __init__(self, output_dir, data_loader, n_words, ixtoword, ind2vector, use_sg, dataset):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        #torch.cuda.set_device(cfg.GPU_ID)
        #cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.dataset = dataset
        self.num_batches = len(self.data_loader)
        
        self.ind2obj=ind2vector[0]


        self.ind2vec_inside=ind2vector[1]
        self.ind2vec_outside=ind2vector[2]
        self.ind2vec_left=ind2vector[3]
        self.ind2vec_right=ind2vector[4]
        self.ind2vec_above=ind2vector[5]
        self.ind2vec_below=ind2vector[6]

        self.use_sg=use_sg


    def build_models(self):
        def count_parameters(model):
            total_param = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    num_param = np.prod(param.size())
                    if param.dim() > 1:
                        print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
                    else:
                        print(name, ':', num_param)
                    total_param += num_param
            return total_param

        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.DS_EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.DS_EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()        

        # #######################generator and discriminators############## #
        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from model import D_NET64, D_NET128, D_NET256
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:

        #print('number of trainable parameters  gen =', count_parameters(netG))
        #print('number of trainable parameters  des1 =', count_parameters(netsD[-3]))
        #print('number of trainable parameters  des2 =', count_parameters(netsD[-2]))
        #print('number of trainable parameters  des3 =', count_parameters(netsD[-1]))
        #print('number of trainable parameters  attn =', count_parameters(atten_sg))

        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)



        # ####################### VICTR ############## #
        if self.use_sg:
            print("[VICTR] Initializing VICTR...")
            
            atten_sg=SG_Attention(query_dim=cfg.TEXT.DS_EMBEDDING_DIM, sg_dim=cfg.SG.SG_DIM)
            for p in atten_sg.parameters():
                p.requires_grad = True
            if cfg.TRAIN.SG_ATTN != '':
                state_dict = \
                torch.load(cfg.TRAIN.SG_ATTN, map_location=lambda storage, loc: storage)
                atten_sg.load_state_dict(state_dict)
                #print("[VICTR] Loading attn_sg from: ", cfg.TRAIN.SG_ATTN)

            #[VICR] Basic Graph node embedding
            sg_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2obj))

            #[VICR] Posiitonal Graph node embedding
            inside_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_inside))
            outside_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_outside))
            left_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_left))
            right_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_right))
            above_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_above))
            below_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_below))


        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
            
            if self.use_sg:
                atten_sg.cuda()
                sg_embedding.cuda()

                inside_embedding.cuda()
                outside_embedding.cuda()
                left_embedding.cuda()
                right_embedding.cuda()
                above_embedding.cuda()
                below_embedding.cuda()
        
        if self.use_sg:

            return [text_encoder, image_encoder, netG, netsD, epoch, sg_embedding, atten_sg, \
            inside_embedding, outside_embedding, left_embedding, right_embedding, above_embedding, below_embedding]

        else:
            return [text_encoder, image_encoder, netG, netsD, epoch]
        

    def define_optimizers(self, netG, netsD, atten_sg=None):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(filter(lambda p: p.requires_grad, netsD[i].parameters()),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        
        if self.use_sg:
            optimizerSGAttn = optim.Adam(atten_sg.parameters(),
                                lr=cfg.TRAIN.SGATTN_LR,
                                betas=(0.5, 0.999))
            return optimizerG, optimizersD, optimizerSGAttn

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch, atten_sg=None):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds models.')
        
        if self.use_sg:
            torch.save(atten_sg.state_dict(),
                '%s/attnsg_epoch_%d.pth' % (self.model_dir, epoch))
            print('Save scene graph attention.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, real_image, name='current'):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'% (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)
        #print(real_image.type)

    def train(self):
        #text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        
        if self.use_sg:

            text_encoder, image_encoder, netG, netsD, start_epoch, sg_embedding, atten_sg, \
            inside_embedding, outside_embedding, left_embedding, right_embedding, above_embedding, below_embedding = self.build_models()
            del  self.ind2obj, self.ind2vec_inside, self.ind2vec_outside, self.ind2vec_left, self.ind2vec_right, self.ind2vec_above, self.ind2vec_below

        else:
            text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        
        
        
        if self.use_sg:
            
            optimizerG, optimizersD, optimizerSGAttn = self.define_optimizers(netG, netsD, atten_sg)
        else:
            optimizerG, optimizersD = self.define_optimizers(netG, netsD)
            
        avg_param_G = copy_G_params(netG)
        #optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        
        error_G=dict()
        log_G=dict()
        error_D=dict()
        log_D=dict()
        epoch_time=dict()
        
        
        
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            image_counter=0
            image_saver_counter=0
            image_saved=0
            
            error_G[epoch]=dict()
            log_G[epoch]=dict()
            error_D[epoch]=dict()
            log_D[epoch]=dict()
            epoch_time[epoch]=dict()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()

                if self.use_sg:

                    imgs, captions, cap_lens, \
                    class_ids, keys, sg5_obj, sg5_att, sg5_rel, sg5_obj_count, \
                    sg5_obj_inside,sg5_rel_inside, \
                    sg5_obj_outside,sg5_rel_outside, \
                    sg5_obj_left,sg5_rel_left, \
                    sg5_obj_right,sg5_rel_right, \
                    sg5_obj_above,sg5_rel_above, \
                    sg5_obj_below,sg5_rel_below = prepare_data(data, use_sg=True)


                    #[VICTR] positional graph - inside
                    sg5_obj_inside_embed=inside_embedding(sg5_obj_inside)#[16, 20, 50]

                    sg5_rel_inside_shape=list(sg5_rel_inside.size()) #[16, 20, 20]
                    sg5_rel_inside_embed=inside_embedding(sg5_rel_inside.view(-1, sg5_rel_inside_shape[1]*sg5_rel_inside_shape[2]))
                    sg5_rel_inside_embed=sg5_rel_inside_embed.view(sg5_rel_inside_shape[0], sg5_rel_inside_shape[1], sg5_rel_inside_shape[2], -1) #[16, 20, 20, 50]              
                    rel_inside_embed_pro=torch.mean(sg5_rel_inside_embed, -2) #[16, 20, 50]

                    #[VICTR] positional graph - outside
                    sg5_obj_outside_embed=outside_embedding(sg5_obj_outside)

                    sg5_rel_outside_shape=list(sg5_rel_outside.size()) #[16, 20, 20]
                    sg5_rel_outside_embed=outside_embedding(sg5_rel_outside.view(-1, sg5_rel_outside_shape[1]*sg5_rel_outside_shape[2]))
                    sg5_rel_outside_embed=sg5_rel_outside_embed.view(sg5_rel_outside_shape[0], sg5_rel_outside_shape[1], sg5_rel_outside_shape[2], -1)
                    rel_outside_embed_pro=torch.mean(sg5_rel_outside_embed, -2)


                    #[VICTR] positional graph - left
                    sg5_obj_left_embed=left_embedding(sg5_obj_left)

                    sg5_rel_left_shape=list(sg5_rel_left.size()) 
                    sg5_rel_left_embed=left_embedding(sg5_rel_left.view(-1, sg5_rel_left_shape[1]*sg5_rel_left_shape[2]))
                    sg5_rel_left_embed=sg5_rel_left_embed.view(sg5_rel_left_shape[0], sg5_rel_left_shape[1], sg5_rel_left_shape[2], -1)
                    rel_left_embed_pro=torch.mean(sg5_rel_left_embed, -2)


                    #[VICTR] positional graph - right
                    sg5_obj_right_embed=right_embedding(sg5_obj_right)

                    sg5_rel_right_shape=list(sg5_rel_right.size()) 
                    sg5_rel_right_embed=right_embedding(sg5_rel_right.view(-1, sg5_rel_right_shape[1]*sg5_rel_right_shape[2]))
                    sg5_rel_right_embed=sg5_rel_right_embed.view(sg5_rel_right_shape[0], sg5_rel_right_shape[1], sg5_rel_right_shape[2], -1)
                    rel_right_embed_pro=torch.mean(sg5_rel_right_embed, -2)


                    #[VICTR] positional graph - above
                    sg5_obj_above_embed=above_embedding(sg5_obj_above)

                    sg5_rel_above_shape=list(sg5_rel_above.size()) 
                    sg5_rel_above_embed=above_embedding(sg5_rel_above.view(-1, sg5_rel_above_shape[1]*sg5_rel_above_shape[2]))
                    sg5_rel_above_embed=sg5_rel_above_embed.view(sg5_rel_above_shape[0], sg5_rel_above_shape[1], sg5_rel_above_shape[2], -1)
                    rel_above_embed_pro=torch.mean(sg5_rel_above_embed, -2)


                    #[VICTR] positional graph - below
                    sg5_obj_below_embed=below_embedding(sg5_obj_below)
                    sg5_rel_below_shape=list(sg5_rel_below.size()) 
                    sg5_rel_below_embed=below_embedding(sg5_rel_below.view(-1, sg5_rel_below_shape[1]*sg5_rel_below_shape[2]))
                    sg5_rel_below_embed=sg5_rel_below_embed.view(sg5_rel_below_shape[0], sg5_rel_below_shape[1], sg5_rel_below_shape[2], -1)
                    rel_below_embed_pro=torch.mean(sg5_rel_below_embed, -2)


                    
                    position_obj=torch.cat((sg5_obj_inside_embed, sg5_obj_outside_embed, sg5_obj_left_embed, sg5_obj_right_embed, sg5_obj_above_embed, sg5_obj_below_embed), -1)
                    del sg5_obj_inside_embed, sg5_obj_outside_embed, sg5_obj_left_embed, sg5_obj_right_embed, sg5_obj_above_embed, sg5_obj_below_embed
                    position_rel=torch.cat((rel_inside_embed_pro, rel_outside_embed_pro, rel_left_embed_pro, rel_right_embed_pro, rel_above_embed_pro, rel_below_embed_pro), -1)
                    del rel_inside_embed_pro, rel_outside_embed_pro, rel_left_embed_pro, rel_right_embed_pro, rel_above_embed_pro, rel_below_embed_pro
                    del sg5_rel_inside_embed, sg5_rel_outside_embed, sg5_rel_left_embed, sg5_rel_right_embed, sg5_rel_above_embed, sg5_rel_below_embed

                        

                    sg5_obj_embed=sg_embedding(sg5_obj) #[16, 5, d]

                    sg5_att_shape=list(sg5_att.size())
                    sg5_att_embed=sg_embedding(sg5_att.view(-1, sg5_att_shape[1]*sg5_att_shape[2]))
                    sg5_att_embed=sg5_att_embed.view(sg5_att_shape[0], sg5_att_shape[1], sg5_att_shape[2], -1) #[14, 5, 10, d] 
 
                    sg5_rel_shape=list(sg5_rel.size())
                    sg5_rel_embed=sg_embedding(sg5_rel.view(-1, sg5_rel_shape[1]*sg5_rel_shape[2]))
                    sg5_rel_embed=sg5_rel_embed.view(sg5_rel_shape[0], sg5_rel_shape[1], sg5_rel_shape[2], -1)

                    att_embed_pro=torch.mean(sg5_att_embed, -2) #[14, 5, 300] 

                    rel_embed_pro=torch.mean(sg5_rel_embed, -2) #[14, 5, 300] 


                    sg_obj_all=torch.cat((sg5_obj_embed, position_obj), -1) 
                    del sg5_obj_embed, position_obj
                    sg_rel_all=torch.cat((rel_embed_pro, position_rel), -1) 
                    del rel_embed_pro, position_rel

                    sg_embed=torch.cat((sg_obj_all, sg_rel_all, att_embed_pro), -1) 
                    del sg_obj_all, sg_rel_all, att_embed_pro


                else:
                    imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                


                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                
                if self.use_sg:
                    sg_mask = sg5_obj_count.view(batch_size, -1)
                    sg_mask_valid = (sg_mask==1)
                    sg_mask_valid = sg_mask_valid.cuda()
                    atten_sg.applyMask(sg_mask_valid)
                    sgattn_output, _ = atten_sg(words_embs.transpose(1, 2).contiguous(), sg_embed)
                    words_embs_sg = torch.cat((words_embs, sgattn_output.transpose(1, 2)), dim=1)
                    sgattn_output = torch.sum(sgattn_output, dim=1, keepdim=False)
                    sent_emb_sg = torch.cat((sent_emb, sgattn_output), dim=1)
                    #(batch_size, output_len, dimensions)



                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb_sg, words_embs_sg, mask, cap_lens)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD, log = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb_sg, real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward(retain_graph=True)
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())
                    D_logs += log

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                if self.use_sg:
                    atten_sg.zero_grad()
                
                
                errG_total, G_logs = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, words_embs_sg, sent_emb, sent_emb_sg, match_labels, cap_lens, class_ids)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                
                if self.use_sg:
                    optimizerSGAttn.step()

                
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print('Epoch [{}/{}] Step [{}/{}]'.format(epoch, self.max_epoch, step,
                                                              self.num_batches) + ' ' + D_logs + ' ' + G_logs)
                # save images
                if gen_iterations % 10000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    #self.save_img_results(netG, fixed_noise, sent_emb, words_embs, mask, image_encoder,
                    #                      captions, cap_lens, epoch, imgs[-1], name='average')
                    load_params(netG, backup_para)
                    #
                    #self.save_img_results(netG, fixed_noise, sent_emb,
                    #                       words_embs, mask, image_encoder,
                    #                       captions, cap_lens,
                    #                       epoch, name='current')
                    
                    error_G[epoch][gen_iterations]=errG_total
                    log_G[epoch][gen_iterations]=G_logs

                    error_D[epoch][gen_iterations]=errD_total
                    log_D[epoch][gen_iterations]=D_logs
                
                image_counter+=cfg.TRAIN.BATCH_SIZE
                if image_counter % 100 == 0:
                    print('%d images trained' % image_counter)
                

            end_t = time.time()
            epoch_time[epoch] = end_t - start_t

            print('''[%d/%d] Loss_D: %.2f Loss_G: %.2f Time: %.2fs''' % (
                epoch, self.max_epoch, errD_total.item(), errG_total.item(), end_t - start_t))
            print('-' * 89)
            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                #self.save_model(netG, avg_param_G, netsD, epoch)
                if self.use_sg:
                    self.save_model(netG, avg_param_G, netsD, epoch, atten_sg)
                else:
                    self.save_model(netG, avg_param_G, netsD, epoch)
                    
            with open(self.model_dir+'/log_epoch'+ str(epoch) +'.pickle', 'wb') as f:
                pickle.dump([error_G[epoch], log_G[epoch],
                             error_D[epoch], log_D[epoch], epoch_time[epoch]], f, protocol=2)
                print('Save log to: ', self.model_dir+'/log_epoch' + str(epoch) + '.pickle')
                
        
        with open(self.model_dir+'/log_all.pickle', 'wb') as f:
                pickle.dump([error_G, log_G,
                             error_D, log_D, epoch_time], f, protocol=2)
                print('Save log to: ', self.model_dir+'/log_epoch'+ str(epoch) +'.pickle')


        if self.use_sg:
            self.save_model(netG, avg_param_G, netsD, self.max_epoch, atten_sg)
        else:
            self.save_model(netG, avg_param_G, netsD, self.max_epoch)
            
            

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()

            # load text encoder
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.DS_EMBEDDING_DIM)
            state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            #load image encoder
            image_encoder = CNN_ENCODER(cfg.TEXT.DS_EMBEDDING_DIM)
            img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
            state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            print('Load image encoder from:', img_encoder_path)
            image_encoder = image_encoder.cuda()
            image_encoder.eval()
            
            #VICTR
            if self.use_sg:
                atten_sg=SG_Attention(query_dim=cfg.TEXT.DS_EMBEDDING_DIM, sg_dim=cfg.SG.SG_DIM)
                state_dict = torch.load(cfg.TRAIN.SG_ATTN, map_location=lambda storage, loc: storage)
                atten_sg.load_state_dict(state_dict)
                atten_sg.eval()
                atten_sg.cuda()
                
                #[VICTR] Basic graph embedding
                sg_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2obj))
                sg_embedding.cuda()
                
                #[VICTR] Positional graph embedding
                inside_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_inside))
                outside_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_outside))
                left_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_left))
                right_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_right))
                above_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_above))
                below_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_below))
                inside_embedding.cuda()
                outside_embedding.cuda()
                left_embedding.cuda()
                right_embedding.cuda()
                above_embedding.cuda()
                below_embedding.cuda()




            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0
            R_count = 0
            R = np.zeros(30000)
            cont = True
            for ii in range(11):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                if (cont == False):
                    break
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if (cont == False):
                        break
                    if step % 100 == 0:
                       print('cnt: ', cnt)
                    # if step > 50:
                    #     break
                    
                    
                    #imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
                    
                    if self.use_sg:
                        imgs, captions, cap_lens, \
                        class_ids, keys, sg5_obj, sg5_att, sg5_rel, sg5_obj_count, \
                        sg5_obj_inside,sg5_rel_inside, \
                        sg5_obj_outside,sg5_rel_outside, \
                        sg5_obj_left,sg5_rel_left, \
                        sg5_obj_right,sg5_rel_right, \
                        sg5_obj_above,sg5_rel_above, \
                        sg5_obj_below,sg5_rel_below = prepare_data(data, use_sg=True)

                        sg5_obj_inside_embed=inside_embedding(sg5_obj_inside)#[16, 20, 50]

                        sg5_rel_inside_shape=list(sg5_rel_inside.size()) #[16, 20, 20]
                        sg5_rel_inside_embed=inside_embedding(sg5_rel_inside.view(-1, sg5_rel_inside_shape[1]*sg5_rel_inside_shape[2]))
                        sg5_rel_inside_embed=sg5_rel_inside_embed.view(sg5_rel_inside_shape[0], sg5_rel_inside_shape[1], sg5_rel_inside_shape[2], -1) #[16, 20, 20, 50]              
                        rel_inside_embed_pro=torch.mean(sg5_rel_inside_embed, -2) #[16, 20, 50]


                        sg5_obj_outside_embed=outside_embedding(sg5_obj_outside)

                        sg5_rel_outside_shape=list(sg5_rel_outside.size()) #[16, 20, 20]
                        sg5_rel_outside_embed=outside_embedding(sg5_rel_outside.view(-1, sg5_rel_outside_shape[1]*sg5_rel_outside_shape[2]))
                        sg5_rel_outside_embed=sg5_rel_outside_embed.view(sg5_rel_outside_shape[0], sg5_rel_outside_shape[1], sg5_rel_outside_shape[2], -1)
                        rel_outside_embed_pro=torch.mean(sg5_rel_outside_embed, -2)



                        sg5_obj_left_embed=left_embedding(sg5_obj_left)

                        sg5_rel_left_shape=list(sg5_rel_left.size()) #[16, 20, 20]
                        sg5_rel_left_embed=left_embedding(sg5_rel_left.view(-1, sg5_rel_left_shape[1]*sg5_rel_left_shape[2]))
                        sg5_rel_left_embed=sg5_rel_left_embed.view(sg5_rel_left_shape[0], sg5_rel_left_shape[1], sg5_rel_left_shape[2], -1)
                        rel_left_embed_pro=torch.mean(sg5_rel_left_embed, -2)



                        sg5_obj_right_embed=right_embedding(sg5_obj_right)

                        sg5_rel_right_shape=list(sg5_rel_right.size()) #[16, 20, 20]
                        sg5_rel_right_embed=right_embedding(sg5_rel_right.view(-1, sg5_rel_right_shape[1]*sg5_rel_right_shape[2]))
                        sg5_rel_right_embed=sg5_rel_right_embed.view(sg5_rel_right_shape[0], sg5_rel_right_shape[1], sg5_rel_right_shape[2], -1)
                        rel_right_embed_pro=torch.mean(sg5_rel_right_embed, -2)


 
                        sg5_obj_above_embed=above_embedding(sg5_obj_above)

                        sg5_rel_above_shape=list(sg5_rel_above.size()) #[16, 20, 20]
                        sg5_rel_above_embed=above_embedding(sg5_rel_above.view(-1, sg5_rel_above_shape[1]*sg5_rel_above_shape[2]))
                        sg5_rel_above_embed=sg5_rel_above_embed.view(sg5_rel_above_shape[0], sg5_rel_above_shape[1], sg5_rel_above_shape[2], -1)
                        rel_above_embed_pro=torch.mean(sg5_rel_above_embed, -2)



                        sg5_obj_below_embed=below_embedding(sg5_obj_below)
                        sg5_rel_below_shape=list(sg5_rel_below.size()) #[16, 20, 20]
                        sg5_rel_below_embed=below_embedding(sg5_rel_below.view(-1, sg5_rel_below_shape[1]*sg5_rel_below_shape[2]))
                        sg5_rel_below_embed=sg5_rel_below_embed.view(sg5_rel_below_shape[0], sg5_rel_below_shape[1], sg5_rel_below_shape[2], -1)
                        rel_below_embed_pro=torch.mean(sg5_rel_below_embed, -2)


                        #[16, 20, d]
                        position_obj=torch.cat((sg5_obj_inside_embed, sg5_obj_outside_embed, sg5_obj_left_embed, sg5_obj_right_embed, sg5_obj_above_embed, sg5_obj_below_embed), -1)
                        del sg5_obj_inside_embed, sg5_obj_outside_embed, sg5_obj_left_embed, sg5_obj_right_embed, sg5_obj_above_embed, sg5_obj_below_embed
                        position_rel=torch.cat((rel_inside_embed_pro, rel_outside_embed_pro, rel_left_embed_pro, rel_right_embed_pro, rel_above_embed_pro, rel_below_embed_pro), -1)
                        del rel_inside_embed_pro, rel_outside_embed_pro, rel_left_embed_pro, rel_right_embed_pro, rel_above_embed_pro, rel_below_embed_pro
                        del sg5_rel_inside_embed, sg5_rel_outside_embed, sg5_rel_left_embed, sg5_rel_right_embed, sg5_rel_above_embed, sg5_rel_below_embed

                            

                        sg5_obj_embed=sg_embedding(sg5_obj) #[16, 20, d]

                        sg5_att_shape=list(sg5_att.size())
                        sg5_att_embed=sg_embedding(sg5_att.view(-1, sg5_att_shape[1]*sg5_att_shape[2]))
                        sg5_att_embed=sg5_att_embed.view(sg5_att_shape[0], sg5_att_shape[1], sg5_att_shape[2], -1) #[14, 20, 10, d] 
     
                        sg5_rel_shape=list(sg5_rel.size())
                        sg5_rel_embed=sg_embedding(sg5_rel.view(-1, sg5_rel_shape[1]*sg5_rel_shape[2]))
                        sg5_rel_embed=sg5_rel_embed.view(sg5_rel_shape[0], sg5_rel_shape[1], sg5_rel_shape[2], -1)

                        att_embed_pro=torch.mean(sg5_att_embed, -2) #[14, 20, 300] 

                        rel_embed_pro=torch.mean(sg5_rel_embed, -2) #[14, 20, 300] 



                        sg_obj_all=torch.cat((sg5_obj_embed, position_obj), -1) 
                        del sg5_obj_embed, position_obj
                        sg_rel_all=torch.cat((rel_embed_pro, position_rel), -1) 
                        del rel_embed_pro, position_rel

                        sg_embed=torch.cat((sg_obj_all, sg_rel_all, att_embed_pro), -1) 
                        del sg_obj_all, sg_rel_all, att_embed_pro
                    else:
                        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
                          
                    
                    
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]
                    
                    if self.use_sg:

                        sg_mask = sg5_obj_count.view(batch_size, -1)
                        sg_mask_valid = (sg_mask==1)
                        sg_mask_valid = sg_mask_valid.cuda()
                        atten_sg.applyMask(sg_mask_valid)

                            
                        sgattn_output, _ = atten_sg(words_embs.transpose(1, 2).contiguous(), sg_embed)

                        words_embs_sg = torch.cat((words_embs, sgattn_output.transpose(1, 2)), dim=1)
                        sgattn_output = torch.sum(sgattn_output, dim=1, keepdim=False)

                        sent_emb_sg = torch.cat((sent_emb, sgattn_output), dim=1)

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb_sg, words_embs_sg, mask, cap_lens)
                    for j in range(batch_size):
                        s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            #print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        # for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d_%d.png' % (s_tmp, k, ii)
                        im.save(fullpath)

                    _, cnn_code = image_encoder(fake_imgs[-1])

                    for i in range(batch_size):
                        mis_captions, mis_captions_len = self.dataset.get_mis_caption(class_ids[i])
                        hidden = text_encoder.init_hidden(99)
                        _, sent_emb_t = text_encoder(mis_captions, mis_captions_len, hidden)
                        rnn_code = torch.cat((sent_emb[i, :].unsqueeze(0), sent_emb_t), 0)
                        ### cnn_code = 1 * nef
                        ### rnn_code = 100 * nef
                        scores = torch.mm(cnn_code[i].unsqueeze(0), rnn_code.transpose(0, 1))  # 1* 100
                        cnn_code_norm = torch.norm(cnn_code[i].unsqueeze(0), 2, dim=1, keepdim=True)
                        rnn_code_norm = torch.norm(rnn_code, 2, dim=1, keepdim=True)
                        norm = torch.mm(cnn_code_norm, rnn_code_norm.transpose(0, 1))
                        scores0 = scores / norm.clamp(min=1e-8)
                        if torch.argmax(scores0) == 0:
                            R[R_count] = 1
                        R_count += 1

                    if R_count >= 30000:
                        sum = np.zeros(10)
                        np.random.shuffle(R)
                        for i in range(10):
                            sum[i] = np.average(R[i * 3000:(i + 1) * 3000 - 1])
                        R_mean = np.average(sum)
                        R_std = np.std(sum)
                        print("R mean:{:.4f} std:{:.4f}".format(R_mean, R_std))
                        cont = False

                    #if cnt >= 30000:
                    #    cont = False






    def analysis(self, dataset, batch_size, save_dir):
        mkdir_p(save_dir)
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.DS_EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # the path to save generated images
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()

            #VICTR
            if self.use_sg:
                print('Load sg encoder from pretrained embedding')
                
                print('Load attn_sg from:', cfg.TRAIN.SG_ATTN)
                atten_sg=SG_Attention(query_dim=cfg.TEXT.DS_EMBEDDING_DIM, sg_dim=cfg.SG.SG_DIM)
                state_dict = torch.load(cfg.TRAIN.SG_ATTN, map_location=lambda storage, loc: storage)
                atten_sg.load_state_dict(state_dict)
                atten_sg.eval()
                atten_sg.cuda()
                
                sg_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2obj))
                sg_embedding.cuda()


                inside_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_inside))
                outside_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_outside))
                left_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_left))
                right_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_right))
                above_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_above))
                below_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.ind2vec_below))
                inside_embedding.cuda()
                outside_embedding.cuda()
                left_embedding.cuda()
                right_embedding.cuda()
                above_embedding.cuda()
                below_embedding.cuda()
                print("position embedding intialized")


            for step, data in enumerate(self.data_loader, 0):
                #cnt += batch_size
                #if (cont == False):
                #    break
                #if step % 100 == 0:
                #   print('cnt: ', cnt)
                # if step > 50:
                #     break

                if self.use_sg:
                    imgs, captions, cap_lens, \
                    class_ids, keys, sg5_obj, sg5_att, sg5_rel, sg5_obj_count, \
                    sg5_obj_inside,sg5_rel_inside, \
                    sg5_obj_outside,sg5_rel_outside, \
                    sg5_obj_left,sg5_rel_left, \
                    sg5_obj_right,sg5_rel_right, \
                    sg5_obj_above,sg5_rel_above, \
                    sg5_obj_below,sg5_rel_below = prepare_data(data, use_sg=True)

                    sg5_obj_inside_embed=inside_embedding(sg5_obj_inside)#[16, 20, 50]

                    sg5_rel_inside_shape=list(sg5_rel_inside.size()) #[16, 20, 20]
                    sg5_rel_inside_embed=inside_embedding(sg5_rel_inside.view(-1, sg5_rel_inside_shape[1]*sg5_rel_inside_shape[2]))
                    sg5_rel_inside_embed=sg5_rel_inside_embed.view(sg5_rel_inside_shape[0], sg5_rel_inside_shape[1], sg5_rel_inside_shape[2], -1) #[16, 20, 20, 50]              
                    rel_inside_embed_pro=torch.mean(sg5_rel_inside_embed, -2) #[16, 20, 50]


                    sg5_obj_outside_embed=outside_embedding(sg5_obj_outside)

                    sg5_rel_outside_shape=list(sg5_rel_outside.size()) #[16, 20, 20]
                    sg5_rel_outside_embed=outside_embedding(sg5_rel_outside.view(-1, sg5_rel_outside_shape[1]*sg5_rel_outside_shape[2]))
                    sg5_rel_outside_embed=sg5_rel_outside_embed.view(sg5_rel_outside_shape[0], sg5_rel_outside_shape[1], sg5_rel_outside_shape[2], -1)
                    rel_outside_embed_pro=torch.mean(sg5_rel_outside_embed, -2)



                    sg5_obj_left_embed=left_embedding(sg5_obj_left)

                    sg5_rel_left_shape=list(sg5_rel_left.size()) #[16, 20, 20]
                    sg5_rel_left_embed=left_embedding(sg5_rel_left.view(-1, sg5_rel_left_shape[1]*sg5_rel_left_shape[2]))
                    sg5_rel_left_embed=sg5_rel_left_embed.view(sg5_rel_left_shape[0], sg5_rel_left_shape[1], sg5_rel_left_shape[2], -1)
                    rel_left_embed_pro=torch.mean(sg5_rel_left_embed, -2)



                    sg5_obj_right_embed=right_embedding(sg5_obj_right)

                    sg5_rel_right_shape=list(sg5_rel_right.size()) #[16, 20, 20]
                    sg5_rel_right_embed=right_embedding(sg5_rel_right.view(-1, sg5_rel_right_shape[1]*sg5_rel_right_shape[2]))
                    sg5_rel_right_embed=sg5_rel_right_embed.view(sg5_rel_right_shape[0], sg5_rel_right_shape[1], sg5_rel_right_shape[2], -1)
                    rel_right_embed_pro=torch.mean(sg5_rel_right_embed, -2)



                    sg5_obj_above_embed=above_embedding(sg5_obj_above)

                    sg5_rel_above_shape=list(sg5_rel_above.size()) #[16, 20, 20]
                    sg5_rel_above_embed=above_embedding(sg5_rel_above.view(-1, sg5_rel_above_shape[1]*sg5_rel_above_shape[2]))
                    sg5_rel_above_embed=sg5_rel_above_embed.view(sg5_rel_above_shape[0], sg5_rel_above_shape[1], sg5_rel_above_shape[2], -1)
                    rel_above_embed_pro=torch.mean(sg5_rel_above_embed, -2)



                    sg5_obj_below_embed=below_embedding(sg5_obj_below)
                    sg5_rel_below_shape=list(sg5_rel_below.size()) #[16, 20, 20]
                    sg5_rel_below_embed=below_embedding(sg5_rel_below.view(-1, sg5_rel_below_shape[1]*sg5_rel_below_shape[2]))
                    sg5_rel_below_embed=sg5_rel_below_embed.view(sg5_rel_below_shape[0], sg5_rel_below_shape[1], sg5_rel_below_shape[2], -1)
                    rel_below_embed_pro=torch.mean(sg5_rel_below_embed, -2)


                    #[16, 20, d]
                    position_obj=torch.cat((sg5_obj_inside_embed, sg5_obj_outside_embed, sg5_obj_left_embed, sg5_obj_right_embed, sg5_obj_above_embed, sg5_obj_below_embed), -1)
                    del sg5_obj_inside_embed, sg5_obj_outside_embed, sg5_obj_left_embed, sg5_obj_right_embed, sg5_obj_above_embed, sg5_obj_below_embed
                    position_rel=torch.cat((rel_inside_embed_pro, rel_outside_embed_pro, rel_left_embed_pro, rel_right_embed_pro, rel_above_embed_pro, rel_below_embed_pro), -1)
                    del rel_inside_embed_pro, rel_outside_embed_pro, rel_left_embed_pro, rel_right_embed_pro, rel_above_embed_pro, rel_below_embed_pro
                    del sg5_rel_inside_embed, sg5_rel_outside_embed, sg5_rel_left_embed, sg5_rel_right_embed, sg5_rel_above_embed, sg5_rel_below_embed

                        

                    sg5_obj_embed=sg_embedding(sg5_obj) #[16, 20, d]

                    sg5_att_shape=list(sg5_att.size())
                    sg5_att_embed=sg_embedding(sg5_att.view(-1, sg5_att_shape[1]*sg5_att_shape[2]))
                    sg5_att_embed=sg5_att_embed.view(sg5_att_shape[0], sg5_att_shape[1], sg5_att_shape[2], -1) #[14, 20, 10, d] 
 
                    sg5_rel_shape=list(sg5_rel.size())
                    sg5_rel_embed=sg_embedding(sg5_rel.view(-1, sg5_rel_shape[1]*sg5_rel_shape[2]))
                    sg5_rel_embed=sg5_rel_embed.view(sg5_rel_shape[0], sg5_rel_shape[1], sg5_rel_shape[2], -1)

                    att_embed_pro=torch.mean(sg5_att_embed, -2) #[14, 20, 300] 

                    rel_embed_pro=torch.mean(sg5_rel_embed, -2) #[14, 20, 300] 


                    sg_obj_all=torch.cat((sg5_obj_embed, position_obj), -1) 
                    del sg5_obj_embed, position_obj
                    sg_rel_all=torch.cat((rel_embed_pro, position_rel), -1) 
                    del rel_embed_pro, position_rel

                    sg_embed=torch.cat((sg_obj_all, sg_rel_all, att_embed_pro), -1) 
                    del sg_obj_all, sg_rel_all, att_embed_pro
                else:
                    imgs, captions, cap_lens, class_ids, keys = prepare_data(data)




                nz = cfg.GAN.Z_DIM


                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    


                    if self.use_sg:

                        sg_mask = sg5_obj_count.view(batch_size, -1)
                        sg_mask_valid = (sg_mask==1)
                        sg_mask_valid = sg_mask_valid.cuda()
                        atten_sg.applyMask(sg_mask_valid)     
                        sgattn_output, sg_attn_map = atten_sg(words_embs.transpose(1, 2).contiguous(), sg_embed)

                        obj_list_sgs=np.array(sg5_obj.cpu())
                        rel_list_sgs=np.array(sg5_rel.cpu())
                        att_list_sgs=np.array(sg5_att.cpu())
                        mask_list_sgs=np.array(sg5_obj_count.cpu())

                        words_embs_sg = torch.cat((words_embs, sgattn_output.transpose(1, 2)), dim=1)
                        sgattn_output = torch.sum(sgattn_output, dim=1, keepdim=False)
                        sent_emb_sg = torch.cat((sent_emb, sgattn_output), dim=1)
                        
                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb_sg, words_embs_sg, mask, cap_lens)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    
                    most_attended_sg_batch=[]
                    object_name_batch=[]
                    object_rel_att_batch=[]

                    for mask_list, obj_list, rel_list, att_list, sg_map, seq_len in zip(mask_list_sgs, obj_list_sgs, rel_list_sgs, att_list_sgs, sg_attn_map.cpu().data.numpy(), cap_lens_np):
                        no_of_obj=np.sum(list(map(lambda x:x==0, mask_list)))
                        obj_names=[dataset.index2obj[o].encode('utf-8') for o in obj_list[:no_of_obj]]
                        new_rel_list=[np.trim_zeros(o) for o in rel_list[:no_of_obj]]
                        new_att_list=[np.trim_zeros(o) for o in att_list[:no_of_obj]]
 
                        sg_recover={}
                        for i, obj in enumerate(obj_names):
                            sg_recover[obj]={}
                            if (len(new_rel_list[i])>0):
                                sg_recover[obj]['rel']=[dataset.index2rel[o].encode('utf-8') for o in new_rel_list[i]]
                            if (len(new_att_list[i])>0):
                                sg_recover[obj]['att']=[dataset.index2att[o].encode('utf-8') for o in new_att_list[i]]

                        attended_sg=np.sum(sg_map,axis=0)
                        if len(obj_names)<1:
                            most_attended_obj='None'
                            most_attended_obj_info='None'
                        else:
                            most_attended_obj=obj_names[np.argmax(attended_sg[:len(obj_names)])]
                            most_attended_obj_info=(most_attended_obj, sg_recover[most_attended_obj])

                        most_attended_sg_batch.append(most_attended_obj_info)
                        object_name_batch.append(obj_names)
                        object_rel_att_batch.append(sg_recover)



                    for j in range(batch_size):
                        save_name = '%s/%d_s_%s' % (save_dir, i, keys[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):

                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k] #[bs, sequenceL, w, h], 

                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)

                        save_caption=os.path.join(save_dir, 'analysis_caption.txt')
                        with open(save_caption, 'a') as f:
                            for i, sent in enumerate(sentences):
                                
                                f.write(keys[j]+": "+' '.join(sent)+"\n")









