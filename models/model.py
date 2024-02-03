#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet34 import resnet34 
from models.GIE import GIE 
from models.EEM import EEM
from models.GIE_util import knn_one_point

def MCM(in_channel, out_channel):
    return nn.Sequential(
            nn.Conv2d(in_channel, 640, kernel_size=1),
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
            nn.Conv2d(640, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channel, kernel_size=1)
        )


class CEGNet(nn.Module):
    def __init__(self, in_channel=3, strides=[2, 2, 1], pn_conv_channels=[3,32,64,256,512]):
        super(CEGNet, self).__init__()
        self.ft_1 = resnet34(in_channel, strides) 
        self.ft_2 = GIE(pn_conv_channels) 
        self.ft_3 = EEM()  

    def forward(self, rgb , xyz): 
        ft_1 = self.ft_1(rgb)
        b, c, h, w = ft_1.size()
        
        down_xyz = F.interpolate(xyz[:, :], (h, w), mode='nearest')  
        ft_2 = self.ft_2(down_xyz)

        ft_3 = self.ft_3(rgb)  

        ft_c = torch.cat([ft_1,ft_2, ft_3], dim=1)

        return ft_c, down_xyz  

class model(nn.Module):
    def __init__(self, num_class=21):
        super(model, self).__init__()
        self.num_class = num_class

        self.xyznet = CEGNet() 

        self.trans = MCM(1024 + 512 + 512 + 128, 3 * num_class)  

        self.prim_x = MCM(1024 + 512 + 512 + 128, 4 * num_class)

        self.score = MCM(1024 + 512 + 512 + 128, num_class)

    def forward(self, rgb, xyz, cls_ids):

        ft, ft_ds = self.xyznet(rgb , xyz)  
        b, c, h, w = ft.size() 

        px = self.prim_x(ft)
        tx = self.trans(ft)
        sc = F.sigmoid(self.score(ft))

        cls_ids = cls_ids.view(b).long()
        obj_ids = torch.tensor([i for i in range(b)]).long().cuda()

        # flatten
        px = px.view(b, -1, 4, h, w)[obj_ids, cls_ids]
        tx = tx.view(b, -1, 3, h, w)[obj_ids, cls_ids]
        sc = sc.view(b, -1, h, w)[obj_ids, cls_ids]
        del obj_ids

        # pr[bs, 4, h, w], tx[bs, 3, h, w], xyz[bs, 3, h, w]
        tx = tx + ft_ds 

        return {'pred_r': px.contiguous(),
                'pred_t': tx.contiguous(),
                'pred_s': sc.contiguous(),
                'cls_id': cls_ids.contiguous()}

class get_loss(nn.Module):
    def __init__(self, dataset, scoring_weight=0.01, loss_type = "ADD", train = True):

        super(get_loss, self).__init__()
        self.prim_groups = dataset.prim_groups  # [obj_i:[gi:tensor[3, n]]]
        self.sym_list = dataset.get_sym_list()
        self.scoring_weight = scoring_weight
        self.loss_type = loss_type
        self.train = train

        self.select_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    def quaternion_matrix(self, pr):
        R = torch.cat(((1.0 - 2.0 * (pr[2, :] ** 2 + pr[3, :] ** 2)).unsqueeze(dim=1), \
                          (2.0 * pr[1, :] * pr[2, :] - 2.0 * pr[0, :] * pr[3, :]).unsqueeze(dim=1), \
                          (2.0 * pr[0, :] * pr[2, :] + 2.0 * pr[1, :] * pr[3, :]).unsqueeze(dim=1), \
                          (2.0 * pr[1, :] * pr[2, :] + 2.0 * pr[3, :] * pr[0, :]).unsqueeze(dim=1), \
                          (1.0 - 2.0 * (pr[1, :] ** 2 + pr[3, :] ** 2)).unsqueeze(dim=1), \
                          (-2.0 * pr[0, :] * pr[1, :] + 2.0 * pr[2, :] * pr[3, :]).unsqueeze(dim=1), \
                          (-2.0 * pr[0, :] * pr[2, :] + 2.0 * pr[1, :] * pr[3, :]).unsqueeze(dim=1), \
                          (2.0 * pr[0, :] * pr[1, :] + 2.0 * pr[2, :] * pr[3, :]).unsqueeze(dim=1), \
                          (1.0 - 2.0 * (pr[1, :] ** 2 + pr[2, :] ** 2)).unsqueeze(dim=1)),
                         dim=1).contiguous().view(-1, 3, 3) 
        return R



    def calculate_ADD_or_ADDS(self, pred, gt_xyz, cls_id): 

        if cls_id.item() in self.sym_list:

            num_valid, _, num_points = gt_xyz.size()
            inds = knn_one_point(pred.permute(0, 2, 1), gt_xyz.permute(0, 2, 1))  
            inds = inds.view(num_valid, 1, num_points).repeat(1, 3, 1)
            tar_tmp = torch.gather(gt_xyz, 2, inds)  
            add_ij = torch.mean(torch.norm(pred - tar_tmp, dim=1), dim=1) 
        else:
            add_ij = torch.mean(torch.norm(pred - gt_xyz, dim=1), dim=1) 

        return add_ij



    def forward(self, preds, mask, gt_r, gt_t, cls_ids, model_xyz, step=20):


        pred_r = preds['pred_r']
        pred_t = preds['pred_t']
        pred_score = preds['pred_s']

        bs, c, h, w = pred_r.size()
        pred_r = pred_r.view(bs, 4, h, w)
        pred_r = pred_r / torch.norm(pred_r, dim=1, keepdim=True)
        pred_r = pred_r.view(bs, 4, -1)
        pred_t = pred_t.view(bs, 3, -1)
        pred_score = pred_score.view(bs, -1)

    
        cls_ids = cls_ids.view(bs)

        # for one batch
        mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1)

        sub_value = torch.zeros(bs).cuda()
        sub_loss_value = torch.zeros(bs).cuda()


        if self.train:

            for i in range(bs):
                # get mask id
                mk = mask[i].view(-1)
                valid_pixels = mk.nonzero().view(-1)
                num_valid = valid_pixels.size()[0]
                if num_valid < 1:
                    continue
                if num_valid > 20:
                    selected = [i * step for i in range(int(num_valid / step))]
                    valid_pixels = valid_pixels[selected]
                    num_valid = valid_pixels.size()[0]

                # get r, t, s, cls
                pr = pred_r[i][:, valid_pixels]  
                pt = pred_t[i][:, valid_pixels]  
                ps = pred_score[i][valid_pixels] 

                # rotation matrix
                R_pre = self.quaternion_matrix(pr) 

                R_tar = gt_r[i].view(1, 3, 3).repeat(num_valid, 1, 1).contiguous()  
                t_tar = gt_t[i].view(1, 3).repeat(num_valid, 1).contiguous()  


                if self.loss_type == "ADD":

                    # model
                    _, _, num_points = model_xyz.shape
                    # print("model_xyz:", model_xyz[i].shape)

                    md_xyz = torch.Tensor(model_xyz[i]).cuda().view(1, 3, num_points).repeat(num_valid, 1, 1)

                    pt = pt.permute(1, 0).contiguous().unsqueeze(dim=2).repeat(1, 1, num_points)  

                    pred = torch.bmm(R_pre, md_xyz) + pt  # nv, 3, np  

                    t_tar = t_tar.contiguous().unsqueeze(dim=2).repeat(1, 1, num_points) 
                    # print("t_tar1:", t_tar.shape)

                    gt_xyz = torch.bmm(R_tar, md_xyz) + t_tar  # nv, 3, np  

                    # ADD(S)
                    add_ij = self.calculate_ADD_or_ADDS(pred, gt_xyz, cls_ids[i]) 

                    if cls_ids[i] + 1 in self.select_id:

                        sub_value[i] = torch.mean(add_ij)

                        sub_loss_value[i] = torch.mean(add_ij * ps - self.scoring_weight * torch.log(ps))  

                    else:

                        sub_value[i] = torch.mean(add_ij) * 0

                        sub_loss_value[i] = torch.mean(add_ij * ps - self.scoring_weight * torch.log(ps)) * 0  
                else:
                    print("the type of loss is error")


            add = torch.mean(sub_value)
            add_loss = torch.mean(sub_loss_value)  

            loss_dict = {'{}_loss(training)'.format(self.loss_type): add_loss.item(), '{}(training)'.format(self.loss_type): add.item()}


            # ignore the some sample with large outlier  
            add_loss = torch.where(torch.isinf(add_loss), torch.full_like(add_loss, 0), add_loss)
            add_loss = torch.where(torch.isnan(add_loss), torch.full_like(add_loss, 0), add_loss)

            return add_loss, loss_dict



        else:



            add_sub_value = torch.zeros(bs).cuda()
            add_sub_loss_value = torch.zeros(bs).cuda()

            for i in range(bs):

                # get mask id 获取 掩膜id
                mk = mask[i].view(-1)
                valid_pixels = mk.nonzero().view(-1)
                num_valid = valid_pixels.size()[0]
                if num_valid < 1:
                    continue
                if num_valid > 20:
                    selected = [i * step for i in range(int(num_valid / step))]
                    valid_pixels = valid_pixels[selected]
                    num_valid = valid_pixels.size()[0]

                # get r, t, s, cls
                pr = pred_r[i][:, valid_pixels]  # [4, nv]
                pt = pred_t[i][:, valid_pixels]  # [3, nv]
                ps = pred_score[i][valid_pixels]  # [nv]

                # rotation matrix
                R_pre = self.quaternion_matrix(pr)

                R_tar = gt_r[i].view(1, 3, 3).repeat(num_valid, 1, 1).contiguous()  # [nv, 3, 3]
                t_tar = gt_t[i].view(1, 3).repeat(num_valid, 1).contiguous()  # [nv, 3]


                _, _, num_points = model_xyz.shape
                # print("model_xyz:", model_xyz[i].shape)

                md_xyz = torch.Tensor(model_xyz[i]).cuda().view(1, 3, num_points).repeat(num_valid, 1, 1)

                pt = pt.permute(1, 0).contiguous().unsqueeze(dim=2).repeat(1, 1, num_points)  # num_valid, 3, num_points
                pred = torch.bmm(R_pre, md_xyz) + pt  # nv, 3, np

                t_tar = t_tar.contiguous().unsqueeze(dim=2).repeat(1, 1, num_points)
                # print("t_tar1:", t_tar.shape)
                gt_xyz = torch.bmm(R_tar, md_xyz) + t_tar  # nv, 3, np
                # print("gt_xyz:", gt_xyz.shape)

                # ADD(S)
                add_ij = self.calculate_ADD_or_ADDS(pred, gt_xyz, cls_ids[i])
                # print("add_ij:", add_ij)

                add_sub_value[i] = torch.mean(add_ij)
                add_sub_loss_value[i] = torch.mean(add_ij * ps - self.scoring_weight * torch.log(ps))


            add = torch.mean(add_sub_value)
            add_loss = torch.mean(add_sub_loss_value)

            loss_dict = {'add_loss(testing)': add_loss.item(), 'add(testing)': add.item()}      

            return add_loss, loss_dict




