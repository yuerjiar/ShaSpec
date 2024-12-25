import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import f1_score, cohen_kappa_score,recall_score,roc_auc_score,balanced_accuracy_score,confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# from Dataset import GAMMA_sub1_dataset
from data import GAMMA_sub1_dataset,OLIVES_dataset,OLIVES_dataset_v1
from model.model import Res_two_branch_disentangle,Res_two_branch_disentangle_0304
from loss import MyLoss,MyLoss2,MyLoss_0304
from torchvision import transforms
import time
import logging
import torch.optim as optim
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline
import argparse
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
         
local_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

#固定随机种子
def set_seed(seed: int=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)                                                                                                                                            
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# set_seed(2024)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(epoch,train_loader,model):
    model.train()
    loss_meter = AverageMeter()

    loss_function1 = MyLoss() 
    # loss_function1 = MyLoss_new()  #KL散度
    loss_function9 = MyLoss_0304()
  

    predicted_list = []
    label_list = []

    # loss_list = []
    for batch_idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].cuda())
        target = Variable(target.long().cuda())
        optimizer.zero_grad()
        if args.model_name == 'ResNet_disentangle':       
            x1_fundus,x2_fundus,x1_oct,x2_oct,fundus_sha_spe,oct_sha_spe,fundus_to_oct,oct_to_fundus,output1,output2,output3,output4,output,x1_fundus_proj,x1_oct_proj,aff_fundus,aff_oct = model(data[0],data[1])
            #损失计算
            loss = loss_function1(x1_fundus,x2_fundus,x1_oct,x2_oct,fundus_sha_spe,oct_sha_spe,fundus_to_oct,oct_to_fundus,output1,output2,output3,output4,output,target,x1_fundus_proj,x1_oct_proj,aff_fundus,aff_oct)

        elif args.model_name == 'ResNet_disentangle_0304':
            x1_fundus,x2_fundus,x1_oct,x2_oct,fundus_sha_spe,oct_sha_spe,fundus_to_oct,oct_to_fundus,output1,output2,output3,output4,output,x1_fundus_proj,x1_oct_proj,aff_fundus,aff_oct,output5,output6 = model(data[0],data[1])
            loss = loss_function9(x1_fundus,x2_fundus,x1_oct,x2_oct,fundus_sha_spe,oct_sha_spe,fundus_to_oct,oct_to_fundus,output1,output2,output3,output4,output,x1_fundus_proj,x1_oct_proj,aff_fundus,aff_oct,output5,output6,target)
        for p,l in zip(output.cpu().detach().float().numpy().argmax(1),target.cpu().detach().float().numpy()):
            predicted_list.append(p)
            label_list.append(l)

        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())

    avg_kappa = cohen_kappa_score(predicted_list, label_list)

    if args.dataset == 'OLIVES':
        # print('====> Train loss:{:.4f}'.format(loss_meter))
        return loss_meter
    elif args.dataset =='GAMMA':
        print('====> Train Kappa:{:.4f}'.format(avg_kappa))
        return loss_meter,avg_kappa  #this
        # return loss_meter,avg_kappa,loss_meter5,loss_meter6,loss_meter7,loss_meter8,loss_meter9,loss_meter10,loss_meter11   #0304


def val(current_epoch,val_loader,model,best_kappa,best_acc):
    '''
    Gamma数据集为best_kappa
    OLIVES数据集为best_acc
    '''
    model.eval()
    prediction_list = []
    label_list = []
    correct_list = []
    one_hot_label_list = []
    probability_list = []
    one_hot_probability_list = []
    predicted_list_kappa = []
    label_list_kappa = []

    prediction_list1 = []
    correct_list1 = []
    predicted_list_kappa1=[]
    label_list_kappa1=[]
    probability_list1=[]
    one_hot_probability_list1=[]
    prediction_list2 = []
    correct_list2 = []
    predicted_list_kappa2=[]
    label_list_kappa2=[]
    probability_list2=[]
    one_hot_probability_list2=[]


    loss_meter = AverageMeter()
    loss_function1 = MyLoss()
    loss_function9 = MyLoss_0304()

    correct_num, data_num = 0, 0
    correct_num1=0
    correct_num2=0

    # correct_num_missing_oct = 0
    # correct_num_missing_fundus = 0
    for batch_idx, (data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
        for m_num in range(len(data)):
            data[m_num] = Variable(data[m_num].float().cuda())

        data_num += target.size(0)
        with torch.no_grad():
            target = Variable(target.long().cuda())
            if args.model_name == 'ResNet_disentangle':   
                x1_fundus,x2_fundus,x1_oct,x2_oct,fundus_sha_spe,oct_sha_spe,fundus_to_oct,oct_to_fundus,output1,output2,output3,output4,output,x1_fundus_proj,x1_oct_proj,aff_fundus,aff_oct= model(data[0],data[1])
                loss = loss_function1(x1_fundus,x2_fundus,x1_oct,x2_oct,fundus_sha_spe,oct_sha_spe,fundus_to_oct,oct_to_fundus,output1,output2,output3,output4,output,target,x1_fundus_proj,x1_oct_proj,aff_fundus,aff_oct)
            elif args.model_name == 'ResNet_disentangle_0304':
                x1_fundus,x2_fundus,x1_oct,x2_oct,fundus_sha_spe,oct_sha_spe,fundus_to_oct,oct_to_fundus,output1,output2,output3,output4,output,x1_fundus_proj,x1_oct_proj,aff_fundus,aff_oct,output5,output6 = model(data[0],data[1])
                loss = loss_function9(x1_fundus,x2_fundus,x1_oct,x2_oct,fundus_sha_spe,oct_sha_spe,fundus_to_oct,oct_to_fundus,output1,output2,output3,output4,output,x1_fundus_proj,x1_oct_proj,aff_fundus,aff_oct,output5,output6,target)

            #output
            _, predicted = torch.max(output.data, 1)
            correct_num += (predicted == target).sum().item()
            correct = (predicted == target)
            loss_meter.update(loss.item())
            prediction_list.append(predicted.cpu().detach().float().numpy())  ##
            label_list.append(target.cpu().detach().float().numpy())  ##
            correct_list.append(correct.cpu().detach().float().numpy())
            one_hot_label_list.append(F.one_hot(target, num_classes=args.num_classes).squeeze(dim=0).cpu().detach().float().numpy())
            probability_list.append(torch.softmax(output, dim=1).cpu().detach().float().numpy()[:,1])  #针对二分类计算AUC需要，GAMMA数据集为三分类任务，不使用probability_list
            one_hot_probability_list.append(torch.softmax(output, dim=1).data.squeeze(dim=0).cpu().detach().float().numpy())  #GAMMA数据集使用one_hot_probability_list
            #与train相同的kappa计算方式
            for p,l in zip(output.cpu().detach().float().numpy().argmax(1),target.cpu().detach().float().numpy()):
                predicted_list_kappa.append(p)
                label_list_kappa.append(l)


            #output1
            _, predicted1 = torch.max(output1.data, 1)
            correct_num1 += (predicted1 == target).sum().item()
            correct1 = (predicted1 == target)
            # loss_meter.update(loss.item())
            prediction_list1.append(predicted1.cpu().detach().float().numpy())  ##
            # label_list.append(target.cpu().detach().float().numpy())  ##
            correct_list1.append(correct1.cpu().detach().float().numpy())
            # one_hot_label_list.append(F.one_hot(target, num_classes=args.num_classes).squeeze(dim=0).cpu().detach().float().numpy())
            probability_list1.append(torch.softmax(output1, dim=1).cpu().detach().float().numpy()[:,1])  #针对二分类计算AUC需要，GAMMA数据集为三分类任务，不使用probability_list
            one_hot_probability_list1.append(torch.softmax(output1, dim=1).data.squeeze(dim=0).cpu().detach().float().numpy())  #GAMMA数据集使用one_hot_probability_list
            #与train相同的kappa计算方式
            for p,l in zip(output1.cpu().detach().float().numpy().argmax(1),target.cpu().detach().float().numpy()):
                predicted_list_kappa1.append(p)
                label_list_kappa1.append(l)
            #output2
            _, predicted2 = torch.max(output2.data, 1)
            correct_num2 += (predicted2 == target).sum().item()
            correct2 = (predicted2 == target)
            # loss_meter.update(loss.item())
            prediction_list2.append(predicted2.cpu().detach().float().numpy())  ##
            # label_list.append(target.cpu().detach().float().numpy())  ##
            correct_list2.append(correct2.cpu().detach().float().numpy())
            # one_hot_label_list.append(F.one_hot(target, num_classes=args.num_classes).squeeze(dim=0).cpu().detach().float().numpy())
            probability_list2.append(torch.softmax(output2, dim=1).cpu().detach().float().numpy()[:,1])  #针对二分类计算AUC需要，GAMMA数据集为三分类任务，不使用probability_list
            one_hot_probability_list2.append(torch.softmax(output2, dim=1).data.squeeze(dim=0).cpu().detach().float().numpy())  #GAMMA数据集使用one_hot_probability_list
            #与train相同的kappa计算方式
            for p,l in zip(output2.cpu().detach().float().numpy().argmax(1),target.cpu().detach().float().numpy()):
                predicted_list_kappa2.append(p)
                label_list_kappa2.append(l)


    aver_acc = correct_num / data_num
    #两种计算Kappa的
    avg_kappa = cohen_kappa_score(prediction_list, label_list)
    avg_kappa_2 = cohen_kappa_score(predicted_list_kappa, label_list_kappa,weights='quadratic')  #this
      
    b_acc = balanced_accuracy_score(y_true=label_list, y_pred=prediction_list)  #另一种计算acc的方式，平衡类别，其他文章中直接用的是aver_acc
    if args.num_classes> 2:
        epoch_auc =roc_auc_score(one_hot_label_list,one_hot_probability_list, multi_class='ovo')

    else:
        epoch_auc = roc_auc_score(label_list,probability_list)
        confusion = confusion_matrix(y_true=label_list, y_pred=prediction_list)
        tp = confusion[1,1]
        tn = confusion[0,0]
        fp = confusion[0,1]
        fn = confusion[1,0]
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

    if args.dataset == 'OLIVES':
  
        print('====> ACC:{:.4f},F1_score:{:.4f},Recall_score:{:.4f},AUC:{:.4f},Sensitivity:{:.4f},Specificity:{:.4f}'.format(aver_acc,F1_Score,Recall_Score,epoch_auc,sensitivity,specificity))
        return aver_acc, epoch_auc,F1_Score, Recall_Score,sensitivity,specificity,F1_Score_w,Recall_Score_w #this

 
    elif args.dataset =='GAMMA':
        print('====> ACC:{:.4f},b_ACC:{:.4f},Kappa:{:.4f},Kappa2:{:.4f},F1_score:{:.4f},Recall_score:{:.4f},AUC:{:.4f}'.format(aver_acc,b_acc,avg_kappa,avg_kappa_2,F1_Score,Recall_Score,epoch_auc))
        kappa = max(avg_kappa,avg_kappa_2)
        if kappa > best_kappa:
            print('new_kappa:{} > best_kappa:{}'.format(kappa, best_kappa))
            best_kappa = kappa
            
            file_name = os.path.join(args.save_dir,
                                 args.model_name + '_' + args.dataset + '_' + args.folder + '_epoch_{}.pth'.format(current_epoch))
            torch.save({
            'epoch': current_epoch,
            'state_dict': model.state_dict(),
            },
            file_name)

        elif aver_acc>=best_acc and aver_acc>=0.65:
            print('new_acc:{} >= best_acc:{}'.format(aver_acc, best_acc))
            best_acc = aver_acc
            file_name = os.path.join(args.save_dir,
                                 args.model_name + '_' + args.dataset + '_' + args.folder + '_epoch_{}.pth'.format(current_epoch))
            torch.save({
            'epoch': current_epoch,
            'state_dict': model.state_dict(),
            },
            file_name)

        return loss_meter.avg, aver_acc, epoch_auc,avg_kappa, F1_Score, Recall_Score,best_kappa,avg_kappa_2,b_acc,best_acc#this
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize',type=int, default=4,help='input batch size of training [default:4]')
    parser.add_argument('--lr',type=float,default=1e-4,help='learning rate')
    parser.add_argument('--dataset',type=str,default='GAMMA')
    parser.add_argument('--start_epoch',type=int,default=1)
    parser.add_argument('--end_epoch',type=int,default=100)
    parser.add_argument('--folder',default='folder0',type=str,help='folder0/folder1/folder2/folder3/folder4')
    parser.add_argument('--model_name',type=str,default='ResNet_disentangle',help='Baseline/ResNet_disentangle/ResNet_disentangle_woaff')
    parser.add_argument('--mode',type=str,default='train')
    parser.add_argument('--save_dir',type=str,default='/media/wenyu/DATA/GAMMA-code/GAMMA_result/model_saved/ourmodel/')
    parser.add_argument('--log_dir',type=str,default='/media/wenyu/DATA/GAMMA-code/GAMMA_result/logs/')
    parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
    parser.add_argument('--time', default=local_time.split(' ')[1], type=str)
    args = parser.parse_args()
    set_seed(2024)
 #数据加载
    if args.dataset == 'GAMMA':

        args.num_classes = 3
        args.modality_name = ['FUN','OCT']
        args.dims = [[(128,256,128)],[(512,512)]]
        args.modality = len(args.dims)
        args.base_path = '/media/wenyu/DATA/Glaucoma_grading/training/'
        args.data_path = '/media/wenyu/DATA/Glaucoma_grading/training/multi-modality_images/'
        filelists = os.listdir(args.data_path)
        kf = KFold(n_splits=5,shuffle=True,random_state=10)
        y=kf.split(filelists)
        count = 0
        train_filelists = [[],[],[],[],[]]
        val_filelists = [[],[],[],[],[]] 
        for tidx,vidx in y:
            train_filelists[count],val_filelists[count] = np.array(filelists)[tidx],np.array(filelists)[vidx]
            count = count + 1
        f_folder = args.folder[-1]
        
        train_dataset = GAMMA_sub1_dataset(dataset_root=args.data_path, 
                                        oct_img_size = args.dims[0],
                                        fundus_img_size= args.dims[1],
                                        mode='train',
                                        label_file=args.base_path + 'glaucoma_grading_training_GT.xlsx',
                                        filelists=np.array(train_filelists[int(f_folder)]))
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batchsize)
        val_dataset = GAMMA_sub1_dataset(dataset_root=args.data_path,  
                                oct_img_size = args.dims[0],
                                fundus_img_size= args.dims[1],
                                mode='val',
                                label_file=args.base_path + 'glaucoma_grading_training_GT.xlsx',
                                filelists=np.array(val_filelists[int(f_folder)]))
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1)
        print('train list:',train_filelists[int(f_folder)])
        print('Val list:',val_filelists[int(f_folder)])

        test_dataset = val_dataset
        test_loader = val_loader
    elif args.dataset == 'OLIVES':
        args.base_path = '/media/wenyu/DATA/OLIVES/'
        args.data_path = '/media/wenyu/DATA/OLIVES/OLIVES_data/'
        args.num_classes = 2
        args.modality_name = ['FUN','OCT']
        args.dims = [[(48,248,248)],[(512,512)]]
        args.modality = len(args.dims)
        filelists = os.listdir(args.data_path)
        kf = KFold(n_splits=5,shuffle=True,random_state=10)
        y=kf.split(filelists)
        count = 0
        train_filelists = [[],[],[],[],[]]
        val_filelists = [[],[],[],[],[]] 

        for tidx,vidx in y:
            train_filelists[count],val_filelists[count] = np.array(filelists)[tidx],np.array(filelists)[vidx]
            count = count + 1
        f_folder = args.folder[-1]
        
        train_dataset = OLIVES_dataset_v1(dataset_root=args.data_path, 
                                        oct_img_size = args.dims[0],
                                        fundus_img_size= args.dims[1],
                                        mode='train',
                                        label_file=args.base_path + 'OLIVES_GT.xlsx',
                                        filelists=np.array(train_filelists[int(f_folder)]))
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batchsize)
        val_dataset = OLIVES_dataset_v1(dataset_root=args.data_path, 
                                        oct_img_size = args.dims[0],
                                        fundus_img_size= args.dims[1],
                                        mode='val',
                                        label_file=args.base_path + 'OLIVES_GT.xlsx',
                                        filelists=np.array(val_filelists[int(f_folder)]))
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1)
        print('train list:',train_filelists[int(f_folder)])
        print('Val list:',val_filelists[int(f_folder)])



    else:
        print('There in no this dataset name')
        raise NameError

    if args.model_name == 'ResNet_disentangle':
        model = Res_two_branch_disentangle()         #模型
        optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-5)
        model.cuda()
        best_kappa = 0.0
        best_acc = 0.0
        loss_list = []
        acc_list = []
        kappa_list = []
        F1_Score_list = []
        Recall_score_list = []
        auc_list = []

        if args.mode =='train':
            path_log = os.path.join(args.log_dir,args.dataset+'_'+args.model_name+'_'+args.folder +'_'+ args.date+'_'+ args.time)
            writer = SummaryWriter(path_log)
            epoch = 0
            print('============Train begin============')
            for epoch in range(args.start_epoch,args.end_epoch + 1):
                if args.dataset == 'GAMMA':
                    print('Epoch {}/{}'.format(epoch,args.end_epoch-1))
                    epoch_loss,avg_kappa = train(epoch,train_loader,model)
                    print("epoch %d avg_loss:%0.3f avg_kappa:%0.4f" % (epoch, epoch_loss.avg,avg_kappa))
                    #loss_meter.avg, aver_acc, avg_kappa, F1_Score, Recall_Score
                    val_loss, acc,epoch_auc, kappa, F1_Score, Recall_Score,best_kappa,avg_kappa_2,b_acc,best_acc= val(epoch,val_loader,model,best_kappa,best_acc)
                    
                    writer.add_scalar('train_loss',epoch_loss.avg,epoch)
                    writer.add_scalar('val_loss:',val_loss,epoch)
                    writer.add_scalar('val_auc:',epoch_auc,epoch)
                    writer.add_scalar('val_kappa:',kappa,epoch)
                    writer.add_scalar('val_kappa2:',avg_kappa_2,epoch)
                    writer.add_scalar('val_acc',acc,epoch)
                elif args.dataset == 'OLIVES':
                    print('Epoch {}/{}'.format(epoch,args.end_epoch-1))
                    epoch_loss= train(epoch,train_loader,model)
                    print("epoch %d avg_loss:%0.3f" % (epoch, epoch_loss.avg))
                    acc, epoch_auc,F1_Score, Recall_Score,sensitivity,specificity,F1_Score_w,Recall_Score_w = val(epoch,val_loader,model,best_acc)
                    writer.add_scalar('train_loss',epoch_loss.avg,epoch)
                    writer.add_scalar('val_sensitivity',sensitivity,epoch) #train loss
                    writer.add_scalar('val_specificity:',specificity,epoch)
                    writer.add_scalar('val_auc:',epoch_auc,epoch)
                    writer.add_scalar('val_acc',acc,epoch)

    elif args.model_name == 'ResNet_disentangle_0304' :
        model = Res_two_branch_disentangle_0304()      
        optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-5)
        model.cuda()
        best_kappa = 0.0
        best_acc = 0.0
        loss_list = []
        acc_list = []
        kappa_list = []
        F1_Score_list = []
        Recall_score_list = []
        auc_list = []

        if args.mode =='train':
            path_log = os.path.join(args.log_dir,args.dataset+'_'+args.model_name+'_'+args.folder +'_'+ args.date+'_'+ args.time)
            writer = SummaryWriter(path_log)
            epoch = 0
            print('============Train begin============')
            for epoch in range(args.start_epoch,args.end_epoch + 1):
                if args.dataset == 'GAMMA':
                    print('Epoch {}/{}'.format(epoch,args.end_epoch-1))
                    # epoch_loss,avg_kappa = train(epoch,train_loader,model)
                    epoch_loss,avg_kappa,loss_meter5,loss_meter6,loss_meter7,loss_meter8,loss_meter9,loss_meter10,loss_meter11 = train(epoch,train_loader,model)
                    
                    print("epoch %d avg_loss:%0.3f avg_kappa:%0.4f" % (epoch, epoch_loss.avg,avg_kappa))
                    #loss_meter.avg, aver_acc, avg_kappa, F1_Score, Recall_Score
                    # val_loss, acc,epoch_auc, kappa, F1_Score, Recall_Score,best_kappa,avg_kappa_2,b_acc= val(epoch,val_loader,model,best_kappa)
                    val_loss, aver_acc,aver_acc5,aver_acc6,avg_kappa_2,avg_kappa_25,avg_kappa_26,epoch_auc,epoch_auc5,epoch_auc6,aver_acc1,aver_acc2,avg_kappa_21,avg_kappa_22,epoch_auc1,epoch_auc2 = val(epoch,val_loader,model,best_kappa,best_acc)                   
                    writer.add_scalar('train_loss',epoch_loss.avg,epoch)
                    writer.add_scalar('val_loss:',val_loss,epoch)
                    writer.add_scalar('val_auc:',epoch_auc,epoch)
                    writer.add_scalar('val_kappa2:',avg_kappa_2,epoch)
                    writer.add_scalar('val_acc',aver_acc,epoch)

                elif args.dataset == 'OLIVES':
                    print('Epoch {}/{}'.format(epoch,args.end_epoch-1))
                    epoch_loss= train(epoch,train_loader,model)
                    print("epoch %d avg_loss:%0.3f" % (epoch, epoch_loss.avg))
                    #loss_meter.avg, aver_acc, avg_kappa, F1_Score, Recall_Score
                    # val_loss, acc,epoch_auc, F1_Score, Recall_Score,best_acc = val(epoch,val_loader,model,best_acc)
                    acc, epoch_auc,F1_Score, Recall_Score,sensitivity,specificity,F1_Score_w,Recall_Score_w, acc5, epoch_auc5,F1_Score5, Recall_Score5,sensitivity5,specificity5,F1_Score_w5,Recall_Score_w5, acc6, epoch_auc6,F1_Score6, Recall_Score6,sensitivity6,specificity6,F1_Score_w6,Recall_Score_w6=val(epoch,val_loader,model,best_kappa,best_acc)
                    writer.add_scalar('train_loss',epoch_loss.avg,epoch) #train loss
                    writer.add_scalar('val_auc:',epoch_auc,epoch)
                    writer.add_scalar('val_acc',acc,epoch)
   
            

    else:
        print('There in no this model name')
        raise NameError
    













    
