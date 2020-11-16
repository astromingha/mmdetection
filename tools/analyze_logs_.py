log_dir = '/home/user/Work/mmdet_newver/tools/logs/20201114_182024.log.json'#'logs_aug/20200416_122849.log'#'logs_aug/20200415_020623.log'#'logs/20200411_094932.log'
train_iter_max = 1250#400 #200(cascade MRCNN)


import matplotlib.pyplot as plt
import json
val_key = {"loss_rpn_cls","loss_rpn_bbox","s0.loss_cls","s0.acc", "s0.loss_bbox", "s0.loss_mask", "s1.loss_cls", "s1.acc",
           "s1.loss_bbox", "s1.loss_mask", "s2.loss_cls", "s2.acc", "s2.loss_bbox", "s2.loss_mask", "loss"}
map_key = {"bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "segm_mAP", "segm_mAP_50"}

col = 1

fig = plt.figure()
for valkey in val_key:
    metric_train, metric_val = [], []
    with open(log_dir,'r') as js:
        lines_all = js.readlines()

        for line in lines_all:
            line_dict = json.loads(line)
            if "mode" in line_dict:
                if line_dict["mode"] == "train":
                    if line_dict["iter"] == train_iter_max:
                        metric_train.append(line_dict[valkey])
                if line_dict["mode"] == "val":
                    if valkey in line_dict:
                        metric_val.append(line_dict[valkey])

    # ax = fig.add_subplot(len(val_key),col,row)


    # plt.plot(range(len(metric_train)),metric_train,label='train '+valkey)
    # plt.plot(range(len(metric_val)),metric_val,label='val '+valkey)
    # plt.grid(color='gray', dashes=(2, 2))
    # plt.legend(loc='best')
    # plt.savefig('/home/user/Desktop/ndmi_graphs/'+valkey+'.png')
    # plt.close()




for valkey in map_key:
    mAP = []
    with open(log_dir,'r') as js:
        lines_all = js.readlines()

        for line in lines_all:
            line_dict = json.loads(line)
            if "mode" in line_dict:
                if line_dict["mode"] == "val":
                    if valkey in line_dict:
                        mAP.append(line_dict[valkey])

    plt.plot(range(len(mAP)), mAP, label=valkey)
    plt.grid(color='gray', dashes=(2, 2))
    plt.legend(loc='best')
    plt.savefig('/home/user/Desktop/ndmi_graphs/'+ valkey + '.png')
    plt.close()
