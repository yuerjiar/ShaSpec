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
        loss_h = 0.5*loss1 + 0.5*loss2
        #跨模态正则化:正则化学习到的特征
        loss3 = self.L2loss(fundus_sha_spe,oct_to_fundus)
        loss4 = self.L2loss(oct_sha_spe,fundus_to_oct)
        loss_r = 0.5*loss3 + 0.5*loss4
        #循环分类损失
        loss5 = self.CEloss(output1,target)
        loss6 = self.CEloss(output2,target)
        loss7 = self.CEloss(output3,target)
        loss8 = self.CEloss(output4,target)    
        loss_ccls = 0.25*loss5 + 0.25*loss6 + 0.25*loss7 + 0.25*loss8
        #目标分类损失
        loss9 = self.CEloss(output,target)
        loss_cls = loss9  
     
        #对齐亲和图-亲和损失
        loss_align = self.L1loss(x1_fundus_proj,x1_oct_proj)

        aff_fundus = aff_fundus.view(num,2048,-1)
        aff_oct = aff_oct.view(num,2048,-1)

        loss_aff = torch.cosine_similarity(aff_fundus,aff_oct,dim=2).mean()

        loss = loss_h + loss_r + loss_ccls + loss_cls + 0.1*loss_align+0.1*loss_aff


        return loss
    
    def hsic(self,x, y):
        '''
        独立性指标,共享特征和特定特征保持独立,最小化hsic
        x和y是一维张量,[512],需根据实际输入形式修改
        '''
        Kx = x.unsqueeze(0) - x.unsqueeze(1) 
        Kx = torch.exp(- Kx** 2) # 计算核矩阵
        Ky = y.unsqueeze(0) - y.unsqueeze(1) 
        Ky = torch.exp(- Ky** 2) # 计算核矩阵
        Kxy = torch.mm(Kx, Ky)
        n = Kxy.shape[0]
        h = torch.trace(Kxy) / n** 2+ torch.mean(Kx) * torch.mean(Ky) - 2* torch.mean(Kxy) / n
        
        return h * n** 2/ (n - 1)** 2
    

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
    



class MyLoss_L2(nn.Module):
    '''
    loss function for our model
    本文模型的损失函数，由正交损失，亲和对齐损失，跨模态翻译损失，组合分类损失以及最后的目标分类损失构成。
    '''
    def __init__(self):
        super(MyLoss_L2, self).__init__()
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
        loss_h = 0.5*loss1 + 0.5*loss2
        #跨模态正则化:正则化学习到的特征
        loss3 = self.L2loss(fundus_sha_spe,oct_to_fundus)
        loss4 = self.L2loss(oct_sha_spe,fundus_to_oct)
        loss_r = 0.5*loss3 + 0.5*loss4
        #循环分类损失
        loss5 = self.CEloss(output1,target)
        loss6 = self.CEloss(output2,target)
        loss7 = self.CEloss(output3,target)
        loss8 = self.CEloss(output4,target)    
        loss_ccls = 0.25*loss5 + 0.25*loss6 + 0.25*loss7 + 0.25*loss8
        #目标分类损失
        loss9 = self.CEloss(output,target)
        loss_cls = loss9  
     
        #对齐亲和图-亲和损失
        loss_align = self.L2loss(x1_fundus_proj,x1_oct_proj)

        aff_fundus = aff_fundus.view(num,2048,-1)
        aff_oct = aff_oct.view(num,2048,-1)

        loss_aff = torch.cosine_similarity(aff_fundus,aff_oct,dim=2).mean()

        loss = loss_h + loss_r + loss_ccls + loss_cls + 0.1*loss_align+0.1*loss_aff


        return loss
    
    def hsic(self,x, y):
        '''
        独立性指标,共享特征和特定特征保持独立,最小化hsic
        x和y是一维张量,[512],需根据实际输入形式修改
        '''
        Kx = x.unsqueeze(0) - x.unsqueeze(1) 
        Kx = torch.exp(- Kx** 2) # 计算核矩阵
        Ky = y.unsqueeze(0) - y.unsqueeze(1) 
        Ky = torch.exp(- Ky** 2) # 计算核矩阵
        Kxy = torch.mm(Kx, Ky)
        n = Kxy.shape[0]
        h = torch.trace(Kxy) / n** 2+ torch.mean(Kx) * torch.mean(Ky) - 2* torch.mean(Kxy) / n
        
        return h * n** 2/ (n - 1)** 2
    

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





class MyLoss_hyper(nn.Module):
    '''
    loss function for our model
    本文模型的损失函数，由正交损失，亲和对齐损失，跨模态翻译损失，组合分类损失以及最后的目标分类损失构成。
    '''
    def __init__(self):
        super(MyLoss_hyper, self).__init__()
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
        loss_h = 0.5*loss1 + 0.5*loss2
        #跨模态正则化:正则化学习到的特征
        loss3 = self.L2loss(fundus_sha_spe,oct_to_fundus)
        loss4 = self.L2loss(oct_sha_spe,fundus_to_oct)
        loss_r = 0.5*loss3 + 0.5*loss4
        #循环分类损失
        loss5 = self.CEloss(output1,target)
        loss6 = self.CEloss(output2,target)
        loss7 = self.CEloss(output3,target)
        loss8 = self.CEloss(output4,target)    
        loss_ccls = 0.25*loss5 + 0.25*loss6 + 0.25*loss7 + 0.25*loss8
        #目标分类损失
        loss9 = self.CEloss(output,target)
        loss_cls = loss9  
     
        #对齐亲和图-亲和损失
        loss_align = self.L1loss(x1_fundus_proj,x1_oct_proj)

        aff_fundus = aff_fundus.view(num,2048,-1)
        aff_oct = aff_oct.view(num,2048,-1)
        loss_aff = torch.cosine_similarity(aff_fundus,aff_oct,dim=2).mean()

        L_inter = loss_r + (loss_align+loss_aff)*0.5

        loss = loss_h + loss_ccls + loss_cls + L_inter


        return loss
    def hsic(self,x, y):
        '''
        独立性指标,共享特征和特定特征保持独立,最小化hsic
        x和y是一维张量,[512],需根据实际输入形式修改
        '''
        Kx = x.unsqueeze(0) - x.unsqueeze(1) 
        Kx = torch.exp(- Kx** 2) # 计算核矩阵
        Ky = y.unsqueeze(0) - y.unsqueeze(1) 
        Ky = torch.exp(- Ky** 2) # 计算核矩阵
        Kxy = torch.mm(Kx, Ky)
        n = Kxy.shape[0]
        h = torch.trace(Kxy) / n** 2+ torch.mean(Kx) * torch.mean(Ky) - 2* torch.mean(Kxy) / n
        
        return h * n** 2/ (n - 1)** 2
    

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


class MyLoss_hyper02(nn.Module):
    '''
    loss function for our model
    本文模型的损失函数，由正交损失，亲和对齐损失，跨模态翻译损失，组合分类损失以及最后的目标分类损失构成。
    '''
    def __init__(self):
        super(MyLoss_hyper02, self).__init__()
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
        loss_h = 0.5*loss1 + 0.5*loss2
        #跨模态正则化:正则化学习到的特征
        loss3 = self.L2loss(fundus_sha_spe,oct_to_fundus)
        loss4 = self.L2loss(oct_sha_spe,fundus_to_oct)
        loss_r = 0.5*loss3 + 0.5*loss4
        #循环分类损失
        loss5 = self.CEloss(output1,target)
        loss6 = self.CEloss(output2,target)
        loss7 = self.CEloss(output3,target)
        loss8 = self.CEloss(output4,target)    
        loss_ccls = 0.25*loss5 + 0.25*loss6 + 0.25*loss7 + 0.25*loss8
        #目标分类损失
        loss9 = self.CEloss(output,target)
        loss_cls = loss9  
     
        #对齐亲和图-亲和损失
        loss_align = self.L1loss(x1_fundus_proj,x1_oct_proj)

        aff_fundus = aff_fundus.view(num,2048,-1)
        aff_oct = aff_oct.view(num,2048,-1)
        loss_aff = torch.cosine_similarity(aff_fundus,aff_oct,dim=2).mean()

        L_inter = loss_r + (loss_align+loss_aff)*0.2

        loss = loss_h + loss_ccls + loss_cls + L_inter


        return loss
    def hsic(self,x, y):
        '''
        独立性指标,共享特征和特定特征保持独立,最小化hsic
        x和y是一维张量,[512],需根据实际输入形式修改
        '''
        Kx = x.unsqueeze(0) - x.unsqueeze(1) 
        Kx = torch.exp(- Kx** 2) # 计算核矩阵
        Ky = y.unsqueeze(0) - y.unsqueeze(1) 
        Ky = torch.exp(- Ky** 2) # 计算核矩阵
        Kxy = torch.mm(Kx, Ky)
        n = Kxy.shape[0]
        h = torch.trace(Kxy) / n** 2+ torch.mean(Kx) * torch.mean(Ky) - 2* torch.mean(Kxy) / n
        
        return h * n** 2/ (n - 1)** 2
    

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


class MyLoss_hyper001(nn.Module):
    '''
    loss function for our model
    本文模型的损失函数，由正交损失，亲和对齐损失，跨模态翻译损失，组合分类损失以及最后的目标分类损失构成。
    '''
    def __init__(self):
        super(MyLoss_hyper001, self).__init__()
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
        loss_h = 0.5*loss1 + 0.5*loss2
        #跨模态正则化:正则化学习到的特征
        loss3 = self.L2loss(fundus_sha_spe,oct_to_fundus)
        loss4 = self.L2loss(oct_sha_spe,fundus_to_oct)
        loss_r = 0.5*loss3 + 0.5*loss4
        #循环分类损失
        loss5 = self.CEloss(output1,target)
        loss6 = self.CEloss(output2,target)
        loss7 = self.CEloss(output3,target)
        loss8 = self.CEloss(output4,target)    
        loss_ccls = 0.25*loss5 + 0.25*loss6 + 0.25*loss7 + 0.25*loss8
        #目标分类损失
        loss9 = self.CEloss(output,target)
        loss_cls = loss9  
     
        #对齐亲和图-亲和损失
        loss_align = self.L1loss(x1_fundus_proj,x1_oct_proj)

        aff_fundus = aff_fundus.view(num,2048,-1)
        aff_oct = aff_oct.view(num,2048,-1)
        loss_aff = torch.cosine_similarity(aff_fundus,aff_oct,dim=2).mean()

        L_inter = loss_r + (loss_align+loss_aff)*0.01

        loss = loss_h + loss_ccls + loss_cls + L_inter


        return loss
    def hsic(self,x, y):
        '''
        独立性指标,共享特征和特定特征保持独立,最小化hsic
        x和y是一维张量,[512],需根据实际输入形式修改
        '''
        Kx = x.unsqueeze(0) - x.unsqueeze(1) 
        Kx = torch.exp(- Kx** 2) # 计算核矩阵
        Ky = y.unsqueeze(0) - y.unsqueeze(1) 
        Ky = torch.exp(- Ky** 2) # 计算核矩阵
        Kxy = torch.mm(Kx, Ky)
        n = Kxy.shape[0]
        h = torch.trace(Kxy) / n** 2+ torch.mean(Kx) * torch.mean(Ky) - 2* torch.mean(Kxy) / n
        
        return h * n** 2/ (n - 1)** 2
    

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




class MyLoss_hyper005(nn.Module):
    '''
    loss function for our model
    本文模型的损失函数，由正交损失，亲和对齐损失，跨模态翻译损失，组合分类损失以及最后的目标分类损失构成。
    '''
    def __init__(self):
        super(MyLoss_hyper005, self).__init__()
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
        loss_h = 0.5*loss1 + 0.5*loss2
        #跨模态正则化:正则化学习到的特征
        loss3 = self.L2loss(fundus_sha_spe,oct_to_fundus)
        loss4 = self.L2loss(oct_sha_spe,fundus_to_oct)
        loss_r = 0.5*loss3 + 0.5*loss4
        #循环分类损失
        loss5 = self.CEloss(output1,target)
        loss6 = self.CEloss(output2,target)
        loss7 = self.CEloss(output3,target)
        loss8 = self.CEloss(output4,target)    
        loss_ccls = 0.25*loss5 + 0.25*loss6 + 0.25*loss7 + 0.25*loss8
        #目标分类损失
        loss9 = self.CEloss(output,target)
        loss_cls = loss9  
     
        #对齐亲和图-亲和损失
        loss_align = self.L1loss(x1_fundus_proj,x1_oct_proj)

        aff_fundus = aff_fundus.view(num,2048,-1)
        aff_oct = aff_oct.view(num,2048,-1)
        loss_aff = torch.cosine_similarity(aff_fundus,aff_oct,dim=2).mean()

        L_inter = loss_r + (loss_align+loss_aff)*0.05

        loss = loss_h + loss_ccls + loss_cls + L_inter


        return loss
    def hsic(self,x, y):
        '''
        独立性指标,共享特征和特定特征保持独立,最小化hsic
        x和y是一维张量,[512],需根据实际输入形式修改
        '''
        Kx = x.unsqueeze(0) - x.unsqueeze(1) 
        Kx = torch.exp(- Kx** 2) # 计算核矩阵
        Ky = y.unsqueeze(0) - y.unsqueeze(1) 
        Ky = torch.exp(- Ky** 2) # 计算核矩阵
        Kxy = torch.mm(Kx, Ky)
        n = Kxy.shape[0]
        h = torch.trace(Kxy) / n** 2+ torch.mean(Kx) * torch.mean(Ky) - 2* torch.mean(Kxy) / n
        
        return h * n** 2/ (n - 1)** 2
    

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


class MyLoss_hyper1(nn.Module):
    '''
    loss function for our model
    本文模型的损失函数，由正交损失，亲和对齐损失，跨模态翻译损失，组合分类损失以及最后的目标分类损失构成。
    '''
    def __init__(self):
        super(MyLoss_hyper1, self).__init__()
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
        loss_h = 0.5*loss1 + 0.5*loss2
        #跨模态正则化:正则化学习到的特征
        loss3 = self.L2loss(fundus_sha_spe,oct_to_fundus)
        loss4 = self.L2loss(oct_sha_spe,fundus_to_oct)
        loss_r = 0.5*loss3 + 0.5*loss4
        #循环分类损失
        loss5 = self.CEloss(output1,target)
        loss6 = self.CEloss(output2,target)
        loss7 = self.CEloss(output3,target)
        loss8 = self.CEloss(output4,target)    
        loss_ccls = 0.25*loss5 + 0.25*loss6 + 0.25*loss7 + 0.25*loss8
        #目标分类损失
        loss9 = self.CEloss(output,target)
        loss_cls = loss9  
     
        #对齐亲和图-亲和损失
        loss_align = self.L1loss(x1_fundus_proj,x1_oct_proj)

        aff_fundus = aff_fundus.view(num,2048,-1)
        aff_oct = aff_oct.view(num,2048,-1)
        loss_aff = torch.cosine_similarity(aff_fundus,aff_oct,dim=2).mean()

        L_inter = loss_r + (loss_align+loss_aff)*1

        loss = loss_h + loss_ccls + loss_cls + L_inter


        return loss
    def hsic(self,x, y):
        '''
        独立性指标,共享特征和特定特征保持独立,最小化hsic
        x和y是一维张量,[512],需根据实际输入形式修改
        '''
        Kx = x.unsqueeze(0) - x.unsqueeze(1) 
        Kx = torch.exp(- Kx** 2) # 计算核矩阵
        Ky = y.unsqueeze(0) - y.unsqueeze(1) 
        Ky = torch.exp(- Ky** 2) # 计算核矩阵
        Kxy = torch.mm(Kx, Ky)
        n = Kxy.shape[0]
        h = torch.trace(Kxy) / n** 2+ torch.mean(Kx) * torch.mean(Ky) - 2* torch.mean(Kxy) / n
        
        return h * n** 2/ (n - 1)** 2
    

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




class MyLoss_0304(nn.Module):
    '''
    loss function for our model
    '''
    def __init__(self):
        super(MyLoss_0304, self).__init__()
        self.L1loss = nn.L1Loss() #L1 损失
        self.L2loss = nn.MSELoss() #L2 损失
        self.CEloss = nn.CrossEntropyLoss()  #交叉熵损失
        # self.KLloss = nn.KLDivLoss(reduction='batchmean')  #KL散度，对xy的输入形式有要求
        

        
    def forward(self, x1_fundus,x2_fundus,x1_oct,x2_oct,fundus_sha_spe,oct_sha_spe,fundus_to_oct,oct_to_fundus,output1,output2,output3,output4,output,x1_fundus_proj,x1_oct_proj,aff_fundus,aff_oct,output5,output6,target):
        
        '''
        output:[batch_size,num_classes]
        '''
        #保证共享特征和特有特在不相关：暂时用的是正交损失，后续试一下正交投影损失
        num = len(x1_fundus)  #batchsize
        loss1 = self.orthogonal_loss(x1_fundus,x2_fundus,num)
        loss2 = self.orthogonal_loss(x1_oct,x2_oct,num)
        loss_h = 0.5*loss1 + 0.5*loss2
        #跨模态正则化:正则化学习到的特征，也可以应对缺失情况：暂定L2 正则化损失
        loss3 = self.L2loss(fundus_sha_spe,oct_to_fundus)
        loss4 = self.L2loss(oct_sha_spe,fundus_to_oct)
        loss_r = 0.5*loss3 + 0.5*loss4
        #循环分类损失
        loss5 = self.CEloss(output1,target)
        loss6 = self.CEloss(output2,target)
        loss7 = self.CEloss(output3,target)
        loss8 = self.CEloss(output4,target)
        loss9 = self.CEloss(output,target)
        loss10 = self.CEloss(output5,target)
        loss11 = self.CEloss(output6,target)
        loss_ccls = 0.25*loss5 + 0.25*loss6 + 0.25*loss7 + 0.25*loss8 + 0.25*loss10 + 0.25*loss11
        loss_cls = loss9
        #对齐：约束x1_fundus,x2_fundus相似。
        #Affinity Mattrix？
        #L2loss，好像一般不用L2损失
        # loss_align = self.L2loss(x1_fundus,x2_fundus)
        #L1loss对齐
        loss_align = self.L1loss(x1_fundus_proj,x1_oct_proj)

        #对齐亲和图
        aff_fundus = aff_fundus.view(num,2048,-1)
        aff_oct = aff_oct.view(num,2048,-1)

        loss_aff = torch.cosine_similarity(aff_fundus,aff_oct,dim=2).mean()




        loss = loss_h + loss_r + loss_ccls + loss_cls + 0.1*loss_align+0.1*loss_aff
            #超参数设置
        # loss =  loss_ccls + loss_cls 

        return loss
    
    def hsic(self,x, y):
        '''
        独立性指标,共享特征和特定特征保持独立,最小化hsic
        x和y是一维张量,[512],需根据实际输入形式修改
        '''
        Kx = x.unsqueeze(0) - x.unsqueeze(1) 
        Kx = torch.exp(- Kx** 2) # 计算核矩阵
        Ky = y.unsqueeze(0) - y.unsqueeze(1) 
        Ky = torch.exp(- Ky** 2) # 计算核矩阵
        Kxy = torch.mm(Kx, Ky)
        n = Kxy.shape[0]
        h = torch.trace(Kxy) / n** 2+ torch.mean(Kx) * torch.mean(Ky) - 2* torch.mean(Kxy) / n
        
        return h * n** 2/ (n - 1)** 2
    

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
    




class MyLoss_new(nn.Module):
    '''
    loss function for our model
    KL散度
    '''
    def __init__(self):
        super(MyLoss_new, self).__init__()
        self.L1loss = nn.L1Loss() #L1 损失
        self.L2loss = nn.MSELoss() #L2 损失
        self.CEloss = nn.CrossEntropyLoss()  #交叉熵损失
        self.KL = nn.KLDivLoss()
        # self.KLloss = nn.KLDivLoss(reduction='batchmean')  #KL散度，对xy的输入形式有要求
        

        
    def forward(self, x1_fundus,x2_fundus,x1_oct,x2_oct,fundus_sha_spe,oct_sha_spe,fundus_to_oct,oct_to_fundus,output1,output2,output3,output4,output,target,x1_fundus_proj,x1_oct_proj,aff_fundus,aff_oct):
        
        '''
        output:[batch_size,num_classes]
        '''
        #保证共享特征和特有特在不相关：暂时用的是正交损失，后续试一下正交投影损失
        num = len(x1_fundus)  #batchsize
        loss1 = self.orthogonal_loss(x1_fundus,x2_fundus,num)
        loss2 = self.orthogonal_loss(x1_oct,x2_oct,num)
        loss_h = 0.5*loss1 + 0.5*loss2
        #跨模态正则化:正则化学习到的特征，也可以应对缺失情况：暂定L2 正则化损失
        loss3 = self.L2loss(fundus_sha_spe,oct_to_fundus)
        loss4 = self.L2loss(oct_sha_spe,fundus_to_oct)
        loss_r = 0.5*loss3 + 0.5*loss4
        #循环分类损失
        loss5 = self.CEloss(output1,target)
        loss6 = self.CEloss(output2,target)
        loss7 = self.CEloss(output3,target)
        loss8 = self.CEloss(output4,target)
        loss9 = self.CEloss(output,target)
        loss_ccls = 0.25*loss5 + 0.25*loss6 + 0.25*loss7 + 0.25*loss8
        loss_cls = loss9
        #对齐：约束x1_fundus,x2_fundus相似。
        #Affinity Mattrix？
        #L2loss，好像一般不用L2损失
        # loss_align = self.L2loss(x1_fundus,x2_fundus)
        #KL散度
        x1_fundus_proj = F.log_softmax(x1_fundus_proj,dim=1)
        x1_oct_proj = F.softmax(x1_oct_proj,dim=1)
        loss_align = self.KL(x1_fundus_proj,x1_oct_proj)

        #对齐亲和图
        aff_fundus = aff_fundus.view(num,2048,-1)
        aff_oct = aff_oct.view(num,2048,-1)

        loss_aff = torch.cosine_similarity(aff_fundus,aff_oct,dim=2).mean()




        loss = loss_h + loss_r + loss_ccls + loss_cls + 0.1*loss_align+0.1*loss_aff
            #超参数设置
        # loss =  loss_ccls + loss_cls 

        return loss
    
    def hsic(self,x, y):
        '''
        独立性指标,共享特征和特定特征保持独立,最小化hsic
        x和y是一维张量,[512],需根据实际输入形式修改
        '''
        Kx = x.unsqueeze(0) - x.unsqueeze(1) 
        Kx = torch.exp(- Kx** 2) # 计算核矩阵
        Ky = y.unsqueeze(0) - y.unsqueeze(1) 
        Ky = torch.exp(- Ky** 2) # 计算核矩阵
        Kxy = torch.mm(Kx, Ky)
        n = Kxy.shape[0]
        h = torch.trace(Kxy) / n** 2+ torch.mean(Kx) * torch.mean(Ky) - 2* torch.mean(Kxy) / n
        
        return h * n** 2/ (n - 1)** 2
    

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
    


class MyLoss2(nn.Module):
    '''
    loss function
    '''
    def __init__(self):
        super(MyLoss2, self).__init__()
        self.L1loss = nn.L1Loss() #L1 损失
        self.L2loss = nn.MSELoss() #L2 损失
        self.CEloss = nn.CrossEntropyLoss()  #交叉熵损失
        # self.KLloss = nn.KLDivLoss(reduction='batchmean')  #KL散度，对xy的输入形式有要求
        

        
    def forward(self, x1_fundus,x2_fundus,x1_oct,x2_oct,fundus_sha_spe,oct_sha_spe,fundus_to_oct,oct_to_fundus,output1,output2,output3,output4,output,target,x1_fundus_proj,x1_oct_proj,aff_fundus,aff_oct):
        
        '''
        output:[batch_size,num_classes]
        '''
        #保证共享特征和特有特在不相关：暂时用的是正交损失，后续试一下正交投影损失
        num = len(x1_fundus)  #batchsize
        # loss1 = self.orthogonal_loss(x1_fundus,x2_fundus,num)
        # loss2 = self.orthogonal_loss(x1_oct,x2_oct,num)
        # loss_h = 0.5*loss1 + 0.5*loss2
        #跨模态正则化:正则化学习到的特征，也可以应对缺失情况：暂定L2 正则化损失
        # loss3 = self.L2loss(fundus_sha_spe,oct_to_fundus)
        # loss4 = self.L2loss(oct_sha_spe,fundus_to_oct)
        # loss_r = 0.5*loss3 + 0.5*loss4
        #循环分类损失
        # loss5 = self.CEloss(output1,target)
        # loss6 = self.CEloss(output2,target)
        # loss7 = self.CEloss(output3,target)
        # loss8 = self.CEloss(output4,target)
        loss9 = self.CEloss(output,target)
        # loss_ccls = 0.25*loss5 + 0.25*loss6 + 0.25*loss7 + 0.25*loss8
        loss_cls = loss9
        #对齐：约束x1_fundus,x2_fundus相似。
        #Affinity Mattrix？
        #L2loss，好像一般不用L2损失
        # loss_align = self.L2loss(x1_fundus,x2_fundus)
        #L1loss对齐
        # loss_align = self.L1loss(x1_fundus_proj,x1_oct_proj)

        # #对齐亲和图
        # aff_fundus = aff_fundus.view(num,2048,-1)
        # aff_oct = aff_oct.view(num,2048,-1)

        # loss_aff = torch.cosine_similarity(aff_fundus,aff_oct,dim=2).mean()




        loss = loss_cls
            #超参数设置
        # loss =  loss_ccls + loss_cls 

        return loss
    
    def hsic(self,x, y):
        '''
        独立性指标,共享特征和特定特征保持独立,最小化hsic
        x和y是一维张量,[512],需根据实际输入形式修改
        '''
        Kx = x.unsqueeze(0) - x.unsqueeze(1) 
        Kx = torch.exp(- Kx** 2) # 计算核矩阵
        Ky = y.unsqueeze(0) - y.unsqueeze(1) 
        Ky = torch.exp(- Ky** 2) # 计算核矩阵
        Kxy = torch.mm(Kx, Ky)
        n = Kxy.shape[0]
        h = torch.trace(Kxy) / n** 2+ torch.mean(Kx) * torch.mean(Ky) - 2* torch.mean(Kxy) / n
        
        return h * n** 2/ (n - 1)** 2
    

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
