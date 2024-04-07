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

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from utils.draw_box_utils import draw_objs


############################################################
# Utils
############################################################

def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model


def get_model_instance_segmentation(num_classes, load_pretrain_weights=True):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=load_pretrain_weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def images_to_video(image_folder, output_video_name, fps=24):
    file_list = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if
                 f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    file_list_sorted = natsort.natsorted(file_list)

    first_img = Image.open(file_list_sorted[0])
    width, height = first_img.size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    for img_path in file_list_sorted:
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out.write(img_rgb)

    out.release()
    print("Video saved successfully.")


def extract_center_line(predict_mask):
    # 使用 Skeletonization 转换为中心线
    skeleton = cv2.ximgproc.thinning(predict_mask)
    # Thresholding to get binary mask
    _, thresh = cv2.threshold(predict_mask, 0.5, 1, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours((thresh * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(max_contour)
    hull = cv2.approxPolyDP(hull, 0.01*cv2.arcLength(hull, True), True)
    M = cv2.moments(hull)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # return hull, (cX, cY)
    return skeleton

def generate_center_line_points(hull, center):
    center_line_points = []
    for i in range(len(hull)):
        point = hull[i][0]
        center_line_points.append(((center[0] + point[0]) // 2, (center[1] + point[1]) // 2))
    return center_line_points


def draw_center_line(image, center_line_points):
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert points to numpy array for drawing line
    points_array = np.array(center_line_points, dtype=np.int32)

    # Draw line
    cv2.polylines(color_img, [points_array], isClosed=False, color=(0, 255, 0), thickness=2)

    # Draw circles at each point
    for point in center_line_points:
        cv2.circle(color_img, tuple(point), 2, (0, 255, 0), -1)

    return color_img


def predict(opt, device):
    assert os.path.exists(opt.class_label), "json file {} dose not exist.".format(opt.class_label)
    with open(opt.class_label, 'r') as json_file:
        categorys = json.load(json_file)

    model = create_model(num_classes=len(categorys), box_thresh=opt.thresh)
    weights_dict = torch.load(opt.weight, map_location=device)
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    file_list = os.listdir(opt.images)
    file_list = natsort.natsorted(file_list)

    model.eval()
    with torch.no_grad():
        for img_name in file_list:
            img_path = os.path.join(opt.images, img_name)
            assert os.path.exists(img_path), f"{img_path} does not exits."
            original_img = Image.open(img_path).convert('RGB')

            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("Inference+NMS Time for {}: {}".format(img_name, t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_masks = predictions["masks"].to("cpu").numpy()
            predict_masks = np.squeeze(predict_masks, axis=1)

            out_dir = os.path.join(opt.runs_dir, 'out')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            imgs_path = os.path.join(out_dir, img_name)
            plot_img = draw_objs(
                original_img, boxes=predict_boxes, classes=predict_classes,
                scores=predict_scores, masks=predict_masks, category_index=categorys,
                line_thickness=2, font='arial.ttf', font_size=100)
            plot_img.save(imgs_path)

            if 2 in predict_classes:
                # Threshold the predict_masks[glue_index] to binary image
                glue_index = np.where(predict_classes == 2)[0]      # Find index of value 2 in predict_classes
                thresholded_predict_mask = (predict_masks[glue_index].squeeze() > 0.5).astype(np.uint8)*255

                # Display thresholded_predict_mask
                cv2.namedWindow('Predict Mask', cv2.WINDOW_NORMAL)
                cv2.imshow('Predict Mask', thresholded_predict_mask)
                cv2.waitKey(10)

                center = extract_center_line(thresholded_predict_mask)
                center_line_points = generate_center_line_points(contour, center)
                # Draw center line
                image_with_center_line = draw_center_line(thresholded_predict_mask, center_line_points)
                cv2.namedWindow('Fit Line', cv2.WINDOW_NORMAL)
                cv2.imshow('Fit Line', image_with_center_line)
                cv2.waitKey(10)
            else:
                continue


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse arguments for model training')
    parser.add_argument('--images', type=str, default='dataset/images', help='COCO dataset root directory')
    parser.add_argument('--runs_dir', type=str, default='runs/exp', help='Train record file directory')
    parser.add_argument('--weight', type=str, default='runs/exp/epoch_models/epoch_model_35.pth',
                        help='Inference with trained weight')
    parser.add_argument('--class_label', type=str, default='cfg/coco_class_idx.json', help='Class label')
    parser.add_argument('--thresh', type=float, default=0.5, help='Model inference thresh')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_arguments()
    print("Predict config is:\n", opt)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    predict(opt, device)
    images_to_video(os.path.join(opt.runs_dir, 'out'), os.path.join(opt.runs_dir, 'demo.mp4'), 5)


