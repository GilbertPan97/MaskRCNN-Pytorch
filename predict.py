import os
import time
import json
import natsort
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from utils.draw_box_utils import draw_objs


############################################################
#  Configurations
############################################################

# project root dir and src root dir
ROOT_DIR = os.path.abspath("./")

# MODEL_WEIGHT_PATH -> path/to/last/trained/model/weight
MODEL_WEIGHT_PATH = os.path.join(ROOT_DIR, "models", "epoch_models", "epoch_model_35.pth")

# predict result save dir
INFERENCE_SAVE_DIR = os.path.join(ROOT_DIR, "out", "predict")

# image src
IMAGE_DIR = os.path.join(ROOT_DIR, "coco_dataset", "images")


class InferenceConfig():
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + sack

    BOX_THRESH = 0.5


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


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


############################################################
# main function
############################################################

def main():
    config = InferenceConfig()
    label_json_path = './record/coco_class_idx.json'

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=config.NUM_CLASSES,
                         box_thresh=config.BOX_THRESH)
    # model = get_model_instance_segmentation(num_classes=config.NUM_CLASSES,
    #                                         load_pretrain_weights=True)

    # load train weights
    assert os.path.exists(MODEL_WEIGHT_PATH), "{} file dose not exist.".format(MODEL_WEIGHT_PATH)
    weights_dict = torch.load(MODEL_WEIGHT_PATH, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # get image files list of IMAGE_DIR
    file_list = os.listdir(IMAGE_DIR)
    file_list = natsort.natsorted(file_list)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        for img_name in file_list:
            # load image
            img_path = os.path.join(IMAGE_DIR, img_name)
            assert os.path.exists(img_path), f"{img_path} does not exits."
            original_img = Image.open(img_path).convert('RGB')

            # from pil image to tensor, do not normalize image
            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)       # expand batch dimension

            # predict and obtain relevant result info
            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_mask = predictions["masks"].to("cpu").numpy()
            predict_mask = np.squeeze(predict_mask, axis=1)     # [batch, 1, h, w] -> [batch, h, w]

            # plot and save predict result
            if not os.path.exists(INFERENCE_SAVE_DIR):
                os.makedirs(INFERENCE_SAVE_DIR)
            imgs_path = os.path.join(INFERENCE_SAVE_DIR, img_name)
            plot_img = draw_objs(
                original_img, boxes=predict_boxes, classes=predict_classes,
                scores=predict_scores, masks=predict_mask, category_index=category_index,
                line_thickness=2, font='arial.ttf', font_size=100)
            plot_img.save(imgs_path)


if __name__ == '__main__':
    main()
