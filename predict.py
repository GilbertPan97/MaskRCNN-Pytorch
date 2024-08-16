import os
import time
import json
import natsort
import numpy as np
import argparse
from PIL import Image
import cv2
import torch
from torchvision import transforms
import torchvision.models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from network_files import MaskRCNN
from utils.draw_box_utils import draw_objs
from network_files.resnet_fpn_model import resnet50_fpn_backbone, resnet101_fpn_backbone


############################################################
# utils
############################################################

# def build_model(num_classes, load_pretrain_weights=True):
#     """
#     Create a Mask R-CNN model with a specified backbone network.
#
#     Parameters:
#     - num_classes (int): Number of classes for the Mask R-CNN model.
#     - load_pretrain_weights (bool): Whether to load pre-trained weights.
#
#     Returns:
#     - model: The initialized Mask R-CNN model.
#     """
#
#     # Use ResNet-50 backbone with FrozenBatchNorm2d
#     backbone = maskrcnn_resnet50_fpn(pretrained=False)
#
#     model = MaskRCNN(backbone, num_classes=num_classes)
#
#     if load_pretrain_weights:
#         # Pre-trained weights URL for COCO: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
#         weights_path = "./weights/maskrcnn_resnet101_fpn_coco.pth"
#         weights_dict = torch.load(weights_path, map_location="cpu")
#
#         # Remove weights for box_predictor and mask_fcn_logits if they exist
#         for k in list(weights_dict.keys()):
#             if ("box_predictor" in k) or ("mask_fcn_logits" in k):
#                 del weights_dict[k]
#
#         print(model.load_state_dict(weights_dict, strict=False))
#
#     return model


def build_model(num_classes, use_resnet101=False, load_pretrain_weights=True):
    """
    Create a Mask R-CNN model with a specified ResNet backbone and FPN.

    Parameters:
    - num_classes (int): Number of classes for the Mask R-CNN model.
    - use_resnet101 (bool): Whether to use ResNet-101 instead of ResNet-50.
    - load_pretrain_weights (bool): Whether to load pre-trained weights.

    Returns:
    - model: The initialized Mask R-CNN model.
    """

    # Choose backbone
    backbone_name = 'resnet101' if use_resnet101 else 'resnet50'
    backbone = resnet_fpn_backbone(backbone_name, pretrained=load_pretrain_weights)
    # The FPN backbone is only used for features, set it to eval mode to avoid issues with BatchNorm layers
    backbone.body.eval()

    # Create the Mask R-CNN model
    model = MaskRCNN(backbone, num_classes=num_classes)

    return model


def build_custom_model(num_classes, use_resnet101=False):
    """
    Build a Mask R-CNN model for instance segmentation with optional ResNet-101 backbone.

    Parameters:
    - num_classes (int): Number of classes for the Mask R-CNN model.
    - use_resnet101 (bool): Whether to use ResNet-101 as the backbone. If False, ResNet-50 is used.
    - load_pretrain_weights (bool): Whether to load pre-trained weights from COCO.

    Returns:
    - model: The initialized Mask R-CNN model.
    """

    if use_resnet101:
        # Use ResNet-101 backbone with FPN
        resnet_backbone = resnet101_fpn_backbone()
    else:
        # Use ResNet-50 backbone with FPN
        resnet_backbone = resnet50_fpn_backbone()

    # Create the Mask R-CNN model with the chosen backbone
    model = MaskRCNN(backbone=resnet_backbone, num_classes=num_classes)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def images_to_video(image_folder, output_video_name, fps=24):
    # Get paths of all image files in the folder
    file_list = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if
                 f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Sort the file list in natural order
    file_list_sorted = natsort.natsorted(file_list)

    # Get width and height of the first image
    first_img = Image.open(file_list_sorted[0])
    width, height = first_img.size

    # Define the video encoder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create the video writing object
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    # Write each frame of the image to the video
    for img_path in file_list_sorted:
        # Read the image
        img = cv2.imread(img_path)
        # Convert BGR format to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Write to the video
        out.write(img_rgb)

    # Release the video writing object
    out.release()
    print("Video saved successfully.")


def predict(opt, device):
    # read class_indict
    assert os.path.exists(opt.class_label), "json file {} dose not exist.".format(opt.class_label)
    with open(opt.class_label, 'r') as json_file:
        categorys = json.load(json_file)

    # create model with backbone: resnet50 or resnet101
    if opt.backbone == 'resnet101':
        model = build_model(num_classes=len(categorys), use_resnet101=True, load_pretrain_weights=False)
    else:
        model = build_model(num_classes=len(categorys), use_resnet101=False, load_pretrain_weights=False)
    model.to(device)

    # load train weights
    assert os.path.exists(opt.weight), "{} file dose not exist.".format(opt.weight)
    weights_dict = torch.load(opt.weight, map_location=device)
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict, strict=False)
    model.to(device)

    # get image files list of IMAGE_DIR
    file_list = os.listdir(opt.images)
    file_list = natsort.natsorted(file_list)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        for img_name in file_list:
            # load image
            img_path = os.path.join(opt.images, img_name)
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
            predict_masks = predictions["masks"].to("cpu").numpy()
            predict_masks = np.squeeze(predict_masks, axis=1)     # [batch, 1, h, w] -> [batch, h, w]

            # plot and save predict result
            out_dir = os.path.join(opt.runs_dir, 'out')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            imgs_path = os.path.join(out_dir, img_name)
            plot_img = draw_objs(
                original_img, boxes=predict_boxes, classes=predict_classes,
                scores=predict_scores, masks=predict_masks, category_index=categorys,
                line_thickness=2, font='arial.ttf', font_size=100)
            plot_img.save(imgs_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse arguments for model training')
    parser.add_argument('--images', type=str, default='dataset/images',
                        help='COCO dataset root directory')

    parser.add_argument('--runs_dir', type=str, default='runs/exp',
                        help='Train record file directory')

    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet101',
                        help='Backbone network to use: resnet50 or resnet101')

    parser.add_argument('--weight', type=str, default='runs/exp/epoch_models/epoch_model.pth',
                        help='Inference with trained weight')

    parser.add_argument('--class_label', type=str, default='cfg/categories.json', help='Class label')

    parser.add_argument('--thresh', type=float, default=0.5, help='Model inference thresh')

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_arguments()
    print("Predict config is:\n", opt)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # inference exec
    predict(opt, device)

    # save with video
    images_to_video(os.path.join(opt.runs_dir, 'out'), os.path.join(opt.runs_dir, 'demo.mp4'), 5)
