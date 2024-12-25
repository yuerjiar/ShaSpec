import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet3D import resnet50,resnet50_disentangle,resnet50_disentangle_v2
from model.resnet2D import res2net50_v1b,res2net50_v1b_disentangle
#resnet和res2net50_v1b分别是2D和3D的ResNet(baseline)
from model.non_local_attention import _NonLocalBlockND

class Res_two_branch_disentangle(nn.Module):
    """
    create a 2-branch-disentangle network
    双分支模型构建
    
    """
    def __init__(self):
        super(Res_two_branch_disentangle, self).__init__()
        self.fundus_branch = res2net50_v1b_disentangle()   #2d分支
        self.oct_branch = resnet50_disentangle_v2()   #3d分支

        #分类器
        self.classifier1 = nn.Sequential(
            nn.Linear(512 * 2, 128),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128,3))
        self.classifier2 = nn.Sequential(
            nn.Linear(512 * 2, 128),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128,3))
        self.classifier3 = nn.Sequential(
            nn.Linear(512 * 2, 128),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128,3))
        self.classifier4 = nn.Sequential(
            nn.Linear(512 * 2, 128),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128,3))
        self.classifier5 = nn.Sequential(
            nn.Linear(512 * 4, 512),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512,3))
        
        #模态翻译模块
        self.modality_translation1 = nn.Sequential(
            nn.Linear(512 * 2,512 * 2),
            nn.Linear(512 * 2,512 * 2),
            nn.ReLU(inplace=True))
        self.modality_translation2 = nn.Sequential(
            nn.Linear(512 * 2,512 * 2),
            nn.Linear(512 * 2,512 * 2),
            nn.ReLU(inplace=True))
        
        self.fc = nn.Linear(512,512)   #将共享特征映射到同一特征空间，再对齐
        self.re1 = nn.ReLU()

        #非局部自注意模块，生成特征亲和图
        self.self_attention = _NonLocalBlockND(in_channels=2048,dimension=2)
        # self.conv_aff = nn.Conv2d(in_channels=2048,out_channels=1,kernel_size=3)
        # self.dropout = nn.Dropout(p=0.2)
   
        # 在oct_branch更改第一个卷积层通道数
        # self.oct_branch.conv1 = nn.Conv2D(256, 64,
        #                                 kernel_size=7,
        #                                 stride=2,
        #                                 padding=3,
        #                                 bias_attr=False)

    def forward(self, fundus_img, oct_img):
        x1_fundus,x2_fundus,xa = self.fundus_branch(fundus_img)
        x1_oct,x2_oct,xb = self.oct_branch(oct_img)
        #xa，xb为特征图。xb是降为之后的特征图，保持与xa维度一致
        #非局部自注意力模块，生成亲和图
        aff_fundus = self.self_attention(xa)
        aff_oct = self.self_attention(xb)
        # aff_fundus = aff_fundus+aff_fundus.permute(0,1,3,2)
        fundus_sha_spe = torch.concat([x1_fundus,x2_fundus], 1)
        oct_sha_spe = torch.concat([x1_oct,x2_oct], 1)
        cross_oct_fundus = torch.concat([x1_oct,x2_fundus], 1)
        cross_fundus_oct = torch.concat([x1_fundus,x2_oct], 1)

        #模态翻译模块
        fundus_to_oct = self.modality_translation1(fundus_sha_spe)  #生成的OCT
        oct_to_fundus = self.modality_translation2(oct_sha_spe)    #生成的fundus
        #所有特征
        all_feature = torch.concat([x1_fundus,x2_fundus,x1_oct,x2_oct], 1)
        
        #映射到同一特征空间
        x1_fundus_proj = self.fc(x1_fundus)
        x1_oct_proj = self.fc(x1_oct)
 
        #全连接+softmax(适用于多分类)
        output1 = self.classifier1(fundus_sha_spe)
        output1 = F.softmax(output1)
        output2 = self.classifier2(oct_sha_spe)
        output2 = F.softmax(output2)
        output3 = self.classifier3(cross_oct_fundus)
        output3 = F.softmax(output3)
        output4 = self.classifier4(cross_fundus_oct)
        output4 = F.softmax(output4)
        output = self.classifier5(all_feature)
        output = F.softmax(output)
 
        return x1_fundus,x2_fundus,x1_oct,x2_oct,fundus_sha_spe,oct_sha_spe,fundus_to_oct,oct_to_fundus,output1,output2,output3,output4,output,x1_fundus_proj,x1_oct_proj,aff_fundus,aff_oct
    
    '''
    x1_fundus和x2_fundus最小化互信息, x1_oct和x2_oct最小化互信息
    x1_fundus和x1_oct对齐
    循环跨模态分类损失 ok
    寻找fundus_sha_spe和oct_sha_spe之间的模态转换关系: 模态翻译模块
    '''

class Res_two_branch_disentangle_0304(nn.Module):
    """
    create a 2-branch-disentangle network
  
    """

    def __init__(self):
        super(Res_two_branch_disentangle_0304, self).__init__()
        self.fundus_branch = res2net50_v1b_disentangle()
        self.oct_branch = resnet50_disentangle_v2()
        self.classifier1 = nn.Sequential(
            nn.Linear(512 * 2, 128),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128,3))
        self.classifier2 = nn.Sequential(
            nn.Linear(512 * 2, 128),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128,3))
        self.classifier3 = nn.Sequential(
            nn.Linear(512 * 2, 128),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128,3))
        self.classifier4 = nn.Sequential(
            nn.Linear(512 * 2, 128),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128,3))
        self.classifier5 = nn.Sequential(
            nn.Linear(512 * 4, 512),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512,3))
        self.classifier6 = nn.Sequential(
            nn.Linear(512 * 2, 128),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128,3))
        self.classifier7 = nn.Sequential(
            nn.Linear(512 * 2, 128),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128,3))
        self.modality_translation1 = nn.Sequential(
            nn.Linear(512 * 2,512 * 2),
            nn.Linear(512 * 2,512 * 2),
            nn.ReLU(inplace=True))
        self.modality_translation2 = nn.Sequential(
            nn.Linear(512 * 2,512 * 2),
            nn.Linear(512 * 2,512 * 2),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(512,512)   #将共享特征映射到同一特征空间，再对齐
        self.re1 = nn.ReLU()
        self.self_attention = _NonLocalBlockND(in_channels=2048,dimension=2)
        # self.conv_aff = nn.Conv2d(in_channels=2048,out_channels=1,kernel_size=3)
        # self.dropout = nn.Dropout(p=0.2)
   
        # 在oct_branch更改第一个卷积层通道数
        # self.oct_branch.conv1 = nn.Conv2D(256, 64,
        #                                 kernel_size=7,
        #                                 stride=2,
        #                                 padding=3,
        #                                 bias_attr=False)

    def forward(self, fundus_img, oct_img):
        x1_fundus,x2_fundus,xa = self.fundus_branch(fundus_img)
        x1_oct,x2_oct,xb = self.oct_branch(oct_img)
        #xa，xb为特征图。xb是降为之后的特征图，保持与xa维度一致
        #非局部自注意力模块，生成亲和图
        aff_fundus = self.self_attention(xa)
        aff_oct = self.self_attention(xb)
        # aff_fundus = aff_fundus+aff_fundus.permute(0,1,3,2)
        fundus_sha_spe = torch.concat([x1_fundus,x2_fundus], 1)
        oct_sha_spe = torch.concat([x1_oct,x2_oct], 1)
        cross_oct_fundus = torch.concat([x1_oct,x2_fundus], 1)
        cross_fundus_oct = torch.concat([x1_fundus,x2_oct], 1)
        shared_feature = torch.concat([x1_fundus,x1_oct],1)
        specific_feature = torch.concat([x2_fundus,x2_oct],1)

        #模态翻译模块
        fundus_to_oct = self.modality_translation1(fundus_sha_spe)  #生成的OCT
        oct_to_fundus = self.modality_translation2(oct_sha_spe)    #生成的fundus
        #所有特征
        all_feature = torch.concat([x1_fundus,x2_fundus,x1_oct,x2_oct], 1)
        
        #映射到同一特征空间
        x1_fundus_proj = self.fc(x1_fundus)
        x1_oct_proj = self.fc(x1_oct)
 
        #全连接+softmax(适用于多分类)
        output1 = self.classifier1(fundus_sha_spe)
        output1 = F.softmax(output1)
        output2 = self.classifier2(oct_sha_spe)
        output2 = F.softmax(output2)
        output3 = self.classifier3(cross_oct_fundus)
        output3 = F.softmax(output3)
        output4 = self.classifier4(cross_fundus_oct)
        output4 = F.softmax(output4)
        output = self.classifier5(all_feature)
        output = F.softmax(output)
        output5 = self.classifier6(shared_feature)
        output5 = F.softmax(output5)
        output6 = self.classifier7(specific_feature)
        output6 = F.softmax(output6)
 
        return x1_fundus,x2_fundus,x1_oct,x2_oct,fundus_sha_spe,oct_sha_spe,fundus_to_oct,oct_to_fundus,output1,output2,output3,output4,output,x1_fundus_proj,x1_oct_proj,aff_fundus,aff_oct,output5,output6






if __name__ == '__main__':
    oct_images = torch.rand(1, 1, 128,256,128).cuda(0)
    fundus_images = torch.rand(1, 3, 224, 224).cuda(0)
 
