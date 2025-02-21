import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pandas
import argparse
import os
import random
import tensorboard_logger as tb_logger
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

select_gene = [
"FLNA",
"MYH9",
"IQGAP1",
"CNOT1",
"MYH10",
"SLC2A1",
"NDUFA10",
"AAAS",
"RPN1",
"DHX9",
"label"]

class CrossAttention(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, hidden_dim):
        super(CrossAttention, self).__init__()
        # Code collation in progress...
    def forward(self, input_a, input_b):
        # Code collation in progress...
        return output_a + output_b

class CNN_GRU_V1(nn.Module):
    def __init__(self,in_channels=1, out_channels=64, batch_size = 64, feature=12,out_class=2):
        super(CNN_GRU_V1, self).__init__()
        self.hint_channel = 64
        self.cross_atten = CrossAttention(input_dim_a=out_channels*feature, 
                                          input_dim_b = out_channels*feature, 
                                          hidden_dim = out_channels*feature)
        # Code collation in progress...
    def forward(self, x):
        # Code collation in progress...
        out = F.softmax(out,dim=1)
        return out

class CsvDataset(Dataset):
    def __init__(self,filepath="",select_data = []):
        df = pandas.read_csv(
                filepath,
                skip_blank_lines = True,
                header=1
                )
        df = df[select_data]
        print(f'the shape of dataframe is {df.shape}')
        feat = df.iloc[:, :-1].astype(float).values
        label = df.iloc[:, -1].astype(float).values
    
        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
from sklearn.model_selection import train_test_split

def creat_loader(batch_size = 64,file_path = '',select_data = [], test_size = 0.3):
    csv_dataset = CsvDataset(file_path,select_data=select_data)
    train_data, test_data = train_test_split(csv_dataset, test_size=test_size, random_state=42)
    train_dataloder = DataLoader(train_data,batch_size = batch_size , shuffle = True)
    test_dataloader =  DataLoader(test_data,batch_size = batch_size , shuffle = True)
    return train_dataloder,test_dataloader

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in args.schedule:
        args.lr = args.lr * args.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def accuracy(output, target, topk=1):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = topk
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # for k in topk:
        correct_k = correct[:topk].float().sum()
        return correct_k.mul_(1.0 / batch_size)
         
def unsqueeze_label(label, class_num,device):  
    batch_size = label.shape[0]
    label_tensor = torch.zeros(batch_size, class_num)
    for i in range(batch_size):
        index = label[i].item()
        label_tensor[int(i), int(index)] = 1
    return label_tensor.cuda(device)
        
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
   
def train(model, train_data, criterion, optimizer,device):
    top1 = AverageMeter()
    train_loss = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    model.train()
    for idx,(batch_x,batch_y) in enumerate(train_data):
        batch_x = batch_x.to(torch.float)
        batch_y = batch_y.to(torch.float)
        batch_x = batch_x.cuda(device)
        batch_y = batch_y.cuda(device)
        label = unsqueeze_label(batch_y, 2, device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, label)
        acc1 = accuracy(outputs, batch_y, topk=(1))
        loss.backward()
        optimizer.step()
        top1.update(acc1, batch_y.size(0))
        train_loss.update(loss)
        #print(loss)
    return top1.avg ,train_loss.avg

def test(model, test_data, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    top1 = AverageMeter()
    test_loss = AverageMeter()
    label_list = []
    predict_list = []
    predict = []
    label_s = []
   
 
    with torch.no_grad():
        for idx,(batch_x,batch_y) in enumerate(test_data):
            label_list.append(batch_y)
            batch_x = batch_x.to(torch.float)
            batch_y = batch_y.to(torch.float)
            batch_x = batch_x.cuda(device)
            batch_y = batch_y.cuda(device)
            outputs = model(batch_x)
            predict_list.append(outputs)
            label = unsqueeze_label(batch_y, 2, device)
            loss = criterion(outputs, label)
            acc1= accuracy(outputs, batch_y, topk=1)
            top1.update(acc1, batch_y.size(0))
            test_loss.update(loss)
    predict_float_list = [tensor.cpu().detach().numpy() for tensor in predict_list]
    for item in predict_float_list:
        for value in item:
            if value[0] > value[1]:
                predict.append(0)
            else:
                predict.append(1)
    for item in label_list:
        for value in item:
            label_s.append(int(value.item())) 
    return top1.avg, test_loss.avg, predict,label_s
def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--test_file_path', default='./data/train/labels_data/UCEC_label.csv')
    parser.add_argument('--best_model_path', default='')
    parser.add_argument('--save_model_path', default='/home/chenkp/CODA/result/UCEC_best_model')
    parser.add_argument('--schedule', default=[10, 20, 30], type=int, nargs='+')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--gpu_id', default='2', type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--trail', default=1, type=int)
    parser.add_argument('--test_size', default=0.3, type=float, help="Slice the dataset and test the ratio of the data occupied.")
    args = parser.parse_args()
    device = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    gpu_id = int(args.gpu_id)
    model = CNN_GRU_V1(feature= 10,out_class=2)
    if args.best_model_path != '':
        print(f'Loading pretrained checkpoint from {args.best_model_path}')
        ckpt = torch.load(args.best_model_path, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        missing_keys, unexpected_keys = \
                model.load_state_dict(ckpt, strict=True)
        if len(missing_keys) != 0:
            print(f'Missing keys in source state dict: {missing_keys}')
        if len(unexpected_keys) != 0:
            print(f'Unexpected keys in source state dict: {unexpected_keys}')

    train_list = nn.ModuleList([])
    train_list.append(model)
    optimizer = optim.SGD(train_list.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    logger = tb_logger.Logger(logdir='{}/tensorboard_{}'.format(args.save_model_path ,args.trail), flush_secs=2)
    model_list = nn.ModuleList([])
    model_list.append(model)
    model_list.cuda()
    L_CE_Loss = nn.CrossEntropyLoss()
    criterion = nn.ModuleList([])
    criterion.append(L_CE_Loss)
    criterion.cuda()
    train_data,test_data = creat_loader(batch_size= args.batch_size,file_path=args.test_file_path,select_data=select_gene, test_size=args.test_size)
   
    best_acc = 0
    save_predict = {}
    save_label = {}
    f = open("{}/best_predict.txt".format(args.save_model_path),'w')
    for epoch in range(args.epoch):
        adjust_learning_rate(optimizer, epoch, args)
        print("lr :{}".format(args.lr))
        train_acc1,train_loss = train(model, train_data, criterion, optimizer, device)
        test_acc1, test_loss, predict, label_s = test(model, test_data, device)
        print('Epoch: {0:} | GPU: {1:} Train_Acc: {2:>2.4f}| Val_Acc: {3:>3.4f}'.format(epoch,gpu_id, train_acc1,test_acc1))
        logger.log_value('train_acc', train_acc1, epoch)
        logger.log_value('test_acc', test_acc1, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        if best_acc<test_acc1:
            best_acc = test_acc1
            best_model = model
            save_predict = predict
            save_label = label_s
    f.write('%s\n %s\n'%(save_predict,save_label)) 
    torch.save(best_model.state_dict(), "{}/best_model_{}".format(args.save_model_path, args.trail))

if __name__ == '__main__':
    main()