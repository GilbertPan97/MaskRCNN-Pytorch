import os
import datetime

import torch
import torchvision.models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from src.utils import transforms
from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from torchvision.ops.misc import FrozenBatchNorm2d
from src.utils.my_dataset_coco import CocoDetection
from utils import train_eval_utils as utils

############################################################
#  Configurations
############################################################

# project root dir and src root dir
ROOT_DIR = os.path.abspath("../")
SRC_ROOT_DIR = os.path.abspath("./")

# train record file dir
LOG_DIR = os.path.join(ROOT_DIR, "log")

# coco dataset root dir(include annotations and images)
DATA_ROOT_DIR = os.path.join(ROOT_DIR, "coco_dataset")

# output model dir(include epoch models and frozen model)
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# train with coco weight or not
TRAIN_WITH_COCO = True

# COCO_MODEL_DIR include model file and coco weight
COCO_MODEL_DIR = os.path.join(SRC_ROOT_DIR, "models")

# if not train with coco weight, this need to be defined
# BASE_WEIGHT_PATH -> path/to/last/trained/model/weight
BASE_WEIGHT_PATH = []


# train configuration
class TrainConfig():
    # Train epochs
    EPOCHS = 36

    # Batch_size
    BatchSize = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + sack

    # Number of workers
    # NUM_WORKERS = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    NUM_WORKERS = 1

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.004
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001


############################################################
# utils
############################################################

def create_model(num_classes, load_pretrain_weights=True):
    # 如果GPU显存很小，batch_size不能设置很大，建议将norm_layer设置成FrozenBatchNorm2d(默认是nn.BatchNorm2d)
    # FrozenBatchNorm2d的功能与BatchNorm2d类似，但参数无法更新
    # trainable_layers include ['layer4', 'layer3', 'layer2', 'layer1', 'conv1']，
    # trainable_layers=5: train all layers
    backbone = resnet50_fpn_backbone(norm_layer=FrozenBatchNorm2d, trainable_layers=5)
    # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth
    # backbone = resnet50_fpn_backbone(pretrain_path="./model/resnet50.pth", trainable_layers=3)
    model = MaskRCNN(backbone, num_classes=num_classes)

    if load_pretrain_weights:
        # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        weights_dict = torch.load("./model/maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
        for k in list(weights_dict.keys()):
            if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
    return model


def get_model_instance_segmentation(num_classes, load_pretrain_weights=True):
    # get maskrcnn model form torchvision.models
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=load_pretrain_weights)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


############################################################
# main function
############################################################

def main():

    # get train config
    config = TrainConfig()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {} device training.".format(device.type))

    # save train info file to det_results_file & seg_results_file
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    det_results_file = LOG_DIR+f"/det_results{now}.txt"
    seg_results_file = LOG_DIR+f"/seg_results{now}.txt"

    # define model input data
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # load train dataset
    train_dataset = CocoDetection(DATA_ROOT_DIR, "train", data_transform["train"])
    print('Using %g dataloader workers' % config.NUM_WORKERS)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=config.BatchSize,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=config.NUM_WORKERS,
                                                    collate_fn=train_dataset.collate_fn)

    # load validation data set
    val_dataset = CocoDetection(DATA_ROOT_DIR, "val", data_transform["val"])
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=config.NUM_WORKERS,
                                                  collate_fn=train_dataset.collate_fn)

    # create model num_classes
    model = create_model(num_classes=config.NUM_CLASSES,
                         load_pretrain_weights=True)
    # model = get_model_instance_segmentation(num_classes=config.NUM_CLASSES,
    #                                         load_pretrain_weights=True)
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=config.LEARNING_RATE,
                                momentum=config.LEARNING_MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[16, 22], gamma=0.1)

    train_loss = []
    learning_rate = []
    val_map = []
    scaler = torch.cuda.amp.GradScaler()
    # if not train with coco weight，train based on BASE_WEIGHT_PATH
    if not TRAIN_WITH_COCO:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(BASE_WEIGHT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(0, config.EPOCHS):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=10,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        det_info, seg_info = utils.evaluate(model, val_data_loader, device=device)

        # write detection into txt (coco id, loss, learning rate)
        with open(det_results_file, "a") as f:
            result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        # write seg into txt (coco id, loss, learning rate)
        with open(seg_results_file, "a") as f:
            result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(det_info[1])         # pascal mAP

        # save epoch model weights
        epoch_dir = os.path.join(MODELS_DIR, "epoch_models")
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, epoch_dir+"/epoch_model_{}.pth".format(epoch))

    # plot loss and lr (learn rate) curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from src.utils.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate, LOG_DIR)

    # plot mAP (Mean Average Precision) curve
    if len(val_map) != 0:
        from src.utils.plot_curve import plot_map
        plot_map(val_map, LOG_DIR)


if __name__ == "__main__":
    main()
