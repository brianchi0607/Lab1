import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision 
from torchvision import datasets
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from torchvision import models
import time
from tqdm import tqdm
import warnings
import copy
from sklearn import metrics
import seaborn as sns
import random
import torch.nn.functional as F
import os
import tensorflow as tf
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from torchsummary import summary
from sklearn.metrics import accuracy_score,classification_report, f1_score,roc_auc_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
#%%
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Some cudnn methods can be random even after fixing the seed 
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

seed_everything(1234)
#%%
def images_transforms(phase):
    if phase == 'training':
        data_transformation =transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomEqualize(10),
            transforms.RandomRotation(degrees=(-25,20)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    else:
        data_transformation=transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
      
    return data_transformation
#%%
def show_CM(validation,prediction):
    matrix = metrics.confusion_matrix(validation,prediction)
    plt.figure(figsize = (6,4))
    sns.heatmap(matrix,cmap = 'coolwarm',linecolor= 'white',
                linewidths= 1,annot= True,fmt = 'd')
    plt.title(' ResNet18 ')
    plt.ylabel('True')
    plt.xlabel('Prediction')
    plt.plot()
    plt.show()
#%%
def imshow(img):
    plt.figure(figsize=(20, 20))
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#%%
class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes =2, size_average=True):
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   
            
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1  
           
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) 

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax
     
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) == focal loss (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss        
#%%
def training(model, train_loader, test_loader, Loss, optimizer, epochs, device, num_class, name):
    model.to(device)
    best_model_wts = None
    best_evaluated_acc = 0
    train_acc = []
    test_acc = []
    test_Recall = []
    test_Precision = []
    test_F1_score = []
    train_loss = []
    test_loss =[]
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer , gamma = 0.96)
    for epoch in range(1, epochs+1):
        with torch.set_grad_enabled(True):
            model.train()
            total_loss=0
            correct=0
            for idx,(data, label) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                        
                data = data.to(device,dtype=torch.float)
                label = label.to(device,dtype=torch.long)

                predict = model(data)      

                loss = Loss(predict, label.squeeze())

                total_loss += loss.item()
                pred = torch.max(predict,1).indices
                correct += pred.eq(label).cpu().sum().item()
                loss.backward()
                optimizer.step()

            total_loss /= len(train_loader.dataset)
            correct = (correct/len(train_loader.dataset))*100.
            print ("Epoch : " , epoch)
            print ("Loss : " , total_loss)
            print ("Train_acc : " , correct)
            train_loss.append(np.mean(total_loss))
            
        scheduler.step()
        accuracy, Recall, Precision, F1_score, test_gt, test_preds = evaluate(model, device, test_loader)
        
        train_acc.append(correct)  
        test_acc.append(accuracy)
        test_Recall.append(Recall)
        test_Precision.append(Precision)
        test_F1_score.append(F1_score)
       
        if accuracy > best_evaluated_acc:
            best_evaluated_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
    #save model
    torch.save(best_model_wts, name+".pt")
    model.load_state_dict(best_model_wts)

    return train_acc , test_acc , test_Recall , test_Precision , test_F1_score, test_gt, test_preds
#%%
def evaluate(model, device, test_loader):
    test_preds = []
    test_gt = []
    correct=0
    TP=0
    TN=0
    FP=0
    FN=0
    with torch.set_grad_enabled(False):
        model.eval()
        for idx,(data,label) in enumerate(test_loader):
            data = data.to(device,dtype=torch.float)
            label = label.to(device,dtype=torch.long)
            predict = model(data)
            pred =  np.argmax(predict.cpu().detach().numpy(), axis=-1)
            y_true = label.cpu().detach().numpy().flatten()
            test_preds = np.concatenate((np.array(test_preds, int), np.array(pred, int)))
            test_gt = np.concatenate((np.array(test_gt, int), np.array(y_true, int)))
            #correct += pred.eq(label).cpu().sum().item()
            for j in range(data.size()[0]):
                #print ("{} pred label: {} ,true label:{}" .format(len(pred),pred[j],int(label[j])))
                if (int (pred[j]) == int (label[j])):
                    correct +=1
                if (int (pred[j]) == 1 and int (label[j]) ==  1):
                    TP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  0):
                    TN += 1
                if (int (pred[j]) == 1 and int (label[j]) ==  0):
                    FP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  1):
                    FN += 1
        print ("TP : " , TP)
        print ("TN : " , TN)
        print ("FP : " , FP)
        print ("FN : " , FN)

        print ("num_correct :",correct ," / " , len(test_loader.dataset))
        Recall = TP/(TP+FN)
        print ("Recall : " ,  Recall )

        Precision = TP/(TP+FP)
        print ("Precision : " ,  Precision )

        F1_score = 2 * Precision * Recall / (Precision + Recall)
        print ("F1 - score : " , F1_score)

        correct = (correct/len(test_loader.dataset))*100.
        print ("Accuracy : " , correct ,"%")

    return correct , Recall , Precision , F1_score, test_gt, test_preds
#%%
if __name__=="__main__":
    IMAGE_SIZE=(128,128)
    batch_size=128
    learning_rate = 0.001
    epochs=30
    num_classes=2

    train_path= r'archive/chest_xray/train'
    test_path= r'archive/chest_xray/test'

    trainset=datasets.ImageFolder(train_path,transform=images_transforms('train'))
    testset=datasets.ImageFolder(test_path,transform=images_transforms('test'))

    train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)
    test_loader = DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=2)

    examples=iter(train_loader)
    images,labels=examples.next()
    print(images.shape)
    imshow(torchvision.utils.make_grid(images[:56],pad_value=20))

    model = torchvision.models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, 2)
    criterion = focal_loss().to(device)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=learning_rate)
    
    print (train_loader)
    dataiter = iter(train_loader)
    images , labels = dataiter.next()
    print (type(images) , type(labels))
    print (images.size(),labels.size())

    Loss = focal_loss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr = learning_rate)
    train_acc , test_acc , test_Recall , test_Precision , test_F1_score, test_gt, test_preds  = training(model, train_loader, test_loader, Loss, optimizer,epochs, device, num_classes, 'CNN_chest')
    
    plt.plot(train_acc, label = 'trian_acc')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(["Accuracy_train"],loc = 'upper right')
    plt.title(' ResNet18 ')
    plt.show()  
    
    plt.plot(test_acc, label = 'test_acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(["Accuracy_test"],loc = 'upper right')
    plt.title(' ResNet18 ')
    plt.show()  
    
    plt.plot(test_F1_score, label = 'test_F1_score')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(["test_F1_score"],loc = 'upper right')
    plt.title(' ResNet18 ')
    plt.show()  
    
    show_CM(test_gt, test_preds)
