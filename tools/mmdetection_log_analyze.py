log_dir = 'logs_aug/20200415_020623.log'
train_iter = 400 #200(cascade MRCNN)
model = "cascade"

import matplotlib.pyplot as plt
import json
epoch_train = []
loss_val=[]
loss_train=[]

if model == "cascade":
    loss_cls = "s2.loss_cls"
    s2_acc = "s2.acc"
else:
    loss_cls = "loss_cls"
    s2_acc = "loss_bbox"


s2_acc_val = []
s2_acc_train = []

s2_loss_cls_val = []
s2_loss_cls_train = []

maxvalue = 0
minvalue = 99999
minvalue_cls = 99999

with open(log_dir+'.json','r') as js:
    lines_train = js.readlines()

    for line in lines_train:
        line_train = json.loads(line)
        if line_train["iter"] == train_iter:
            epoch_train.append(line_train["epoch"])
            loss_train.append(line_train["loss"])
            s2_loss_cls_train.append(line_train[loss_cls])
            s2_acc_train.append(line_train[s2_acc])

start_epoch = epoch_train[0]

with open(log_dir,'r') as f:
    lines_val = f.readlines()
    for line in lines_val:
        if 'Epoch(train)' in line:
            linelist_val = line.split(',')

            for elem in linelist_val:
                if s2_acc in elem:
                    s2_acc_value = float(elem.split(":")[-1])
                    s2_acc_val.append(s2_acc_value)

                    if maxvalue < s2_acc_value:
                        maxvalue = s2_acc_value
                        maxvalue_epoch = len(s2_acc_val)-1 + start_epoch

                elif loss_cls in elem:
                    s2_loss_cls_value = float(elem.split(":")[-1])
                    s2_loss_cls_val.append(s2_loss_cls_value)

                    if minvalue_cls > s2_loss_cls_value:
                        minvalue_cls = s2_loss_cls_value
                        minvalue_cls_epoch = len(s2_loss_cls_val) - 1 + start_epoch


            lossvalue = float(linelist_val[-1].split(":")[-1])
            loss_val.append(lossvalue)

            if minvalue > lossvalue:
                minvalue = lossvalue
                minvalue_epoch = len(loss_val)-1 + start_epoch



print("  train_min : ",min(loss_train),"  val_min : ",min(loss_val),"  epoch :",minvalue_epoch)
print(minvalue," ", minvalue_epoch)
print("clas_loss", minvalue_cls, " ", min(s2_loss_cls_val), "  ",minvalue_cls_epoch)
print("s2_acc",max(s2_acc_val), maxvalue ,maxvalue_epoch)#closer ep35:63.4 ep42:63.1 -> s2.acc closer


fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
# plt.figure(1)
ax1.plot(range(start_epoch,start_epoch+len(s2_acc_val)),s2_acc_val,label='val s2.accuracy')
ax1.plot(range(start_epoch,start_epoch+len(s2_acc_train)),s2_acc_train,label='train s2.accuracy')
ax1.grid(color='gray',dashes=(2,2))
ax1.legend(loc='best')
# plt.figure(2)
ax2.plot(range(start_epoch,start_epoch+len(loss_val)), loss_val,label='loss_val')
ax2.plot(range(start_epoch,start_epoch+len(loss_train)), loss_train,label='loss_train')
ax2.grid(color='gray',dashes=(2,2))
ax2.legend(loc='best')
# plt.figure(3)
ax3.plot(range(start_epoch,start_epoch+len(s2_loss_cls_val)),s2_loss_cls_val,label='loss_class_val')
ax3.plot(range(start_epoch,start_epoch+len(s2_loss_cls_train)),s2_loss_cls_train,label='loss_class_train')
ax3.grid(color='gray',dashes=(2,2))
ax3.legend(loc='best')
plt.show()
