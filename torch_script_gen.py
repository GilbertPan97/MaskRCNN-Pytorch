import os
import torch
from PIL import Image
from torchvision import transforms
from backbone import resnet50_fpn_backbone
from network_files import MaskRCNN

import torchvision.models

from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# project root dir and src root dir
ROOT_DIR = os.path.abspath("./")

# output model dir(include epoch models and frozen model)
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# transform pytorch model path
WEIGHT_PATH = os.path.join(MODELS_DIR, "epoch_models", "epoch_model_35.pth")

# export onnx model dir
ONNX_MODEL_DIR = os.path.join(MODELS_DIR, "onnx_models")

# test model images path
IMG_PATH = os.path.join(ROOT_DIR, "coco_dataset", "images", "1_color.png")


############################################################
# utils
############################################################

def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model


def create_model1(num_classes, load_pretrain_weights=True):
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
        weights_dict = torch.load("./coco/maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
        for k in list(weights_dict.keys()):
            if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
    return model


def create_model2(num_classes, load_pretrain_weights=True):
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


def generate_torch_script(pytorch_model, device, img_file, save_path):
    # input of the model (from pil image to tensor, do not normalize image)
    original_img = Image.open(img_file).convert('RGB')      # load image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0).to(device)        # expand batch dimension to device

    # export the model
    pytorch_model.eval()
    if device.type == 'cpu':
        pytorch_model = pytorch_model.cpu()
    traced_script_module = torch.jit.script(pytorch_model)
    # traced_script_module = torch.jit.trace(pytorch_model, img)
    traced_script_module.save(save_path)


############################################################
# main function
############################################################

def main():
    # parameters initial
    num_classes = 2     # background + sack

    # get devices
    device = torch.device("cpu")    # export onnx model with cpu
    print("using {} device.".format(device))

    # create model & load train weights
    model = create_model2(num_classes=num_classes)

    assert os.path.exists(WEIGHT_PATH), "{} file dose not exist.".format(WEIGHT_PATH)
    weights_dict = torch.load(WEIGHT_PATH, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    assert os.path.exists(IMG_PATH), f"{IMG_PATH} does not exits."
    original_img = Image.open(IMG_PATH).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)

    # export torch script model
    model_save_path = os.path.join(MODELS_DIR, "script_model.pt")
    generate_torch_script(model, device, IMG_PATH, model_save_path)     # export torch script model
    model_script = torch.jit.load(model_save_path)
    model_script.eval()
    with torch.no_grad():
        predictions = model_script([img.to(device)])[1][0]

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        predict_mask = predictions["masks"].to("cpu").numpy()


if __name__ == "__main__":
    main()