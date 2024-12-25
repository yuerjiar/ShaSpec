#所有损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random



class MyLoss(nn.Module):
    '''
    loss function for our model
    本文模型的损失函数，由正交损失，亲和对齐损失，跨模态翻译损失，组合分类损失以及最后的目标分类损失构成。
    '''
    def __init__(self):
        super(MyLoss, self).__init__()
        self.L1loss = nn.L1Loss() #L1 损失
        self.L2loss = nn.MSELoss() #L2 损失
        self.CEloss = nn.CrossEntropyLoss()  #交叉熵损失
        # self.KLloss = nn.KLDivLoss(reduction='batchmean')  #KL散度，对xy的输入形式有要求
           
    def forward(self, x1_fundus,x2_fundus,x1_oct,x2_oct,fundus_sha_spe,oct_sha_spe,fundus_to_oct,oct_to_fundus,output1,output2,output3,output4,output,target,x1_fundus_proj,x1_oct_proj,aff_fundus,aff_oct):
        
        '''
        output:[batch_size,num_classes]
        '''
        #保证共享特征和特有特在不相关：暂时用的是正交损失
        num = len(x1_fundus)  #batchsize
        loss1 = self.orthogonal_loss(x1_fundus,x2_fundus,num)
        loss2 = self.orthogonal_loss(x1_oct,x2_oct,num)
        loss_sep = 0.5*loss1 + 0.5*loss2  #loss_sep

         #对齐亲和图-亲和损失
        loss_align = self.L1loss(x1_fundus_proj,x1_oct_proj)

        aff_fundus = aff_fundus.view(num,2048,-1)
        aff_oct = aff_oct.view(num,2048,-1)

        loss_aff = torch.cosine_similarity(aff_fundus,aff_oct,dim=2).mean()
        
        #跨模态正则化:正则化学习到的特征
        loss3 = self.L2loss(fundus_sha_spe,oct_to_fundus)
        loss4 = self.L2loss(oct_sha_spe,fundus_to_oct)
        loss_r = 0.5*loss3 + 0.5*loss4   #
        loss_inter = loss_r + 0.1(loss_aff+loss_align)
        #循环分类损失
        loss5 = self.CEloss(output1,target)
        loss6 = self.CEloss(output2,target)
        loss7 = self.CEloss(output3,target)
        loss8 = self.CEloss(output4,target)    
        loss_ccls = 0.25*loss5 + 0.25*loss6 + 0.25*loss7 + 0.25*loss8  #losss_ccls
        #目标分类损失
        loss9 = self.CEloss(output,target)
        loss_cls = loss9  
     

        loss = loss_sep +  loss_ccls + loss_cls + loss_inter


        return loss
    


    def orthogonal_loss(self,x,y,batchsize):
        '''
        正交损失
        x,y:[batchsize,feature_dim]
        '''
        loss = 0.0
        for i in range(batchsize):
             
            dot_product = torch.dot(x[i,:],y[i,:])
            norm_x = torch.norm(x[i,:])
            norm_y = torch.norm(y[i,:])
            cos_sim = dot_product/(norm_x * norm_y)
            loss = loss + cos_sim.item() # loss + 1-cos_sim.item()

        return loss/batchsize
    


