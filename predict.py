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

from utils.draw_box_utils import draw_objs

from network_files.model import create_model, build_model, build_custom_model


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
    # if opt.backbone == 'resnet101':
    #     model = build_custom_model(num_classes=len(categorys),
    #                                use_resnet101=True,
    #                                load_pretrain_weights=False)
    # else:
    #     model = build_custom_model(num_classes=len(categorys),
    #                                use_resnet101=False,
    #                                load_pretrain_weights=False)
    if opt.backbone == 'resnet101':
        model = build_model(num_classes=len(categorys),
                            load_pretrain_weights=True,
                            use_resnet101=True)
    else:
        model = build_model(num_classes=len(categorys),
                            load_pretrain_weights=True,
                            use_resnet101=False)
    # model = create_model(num_classes=len(categorys),
    #                      load_pretrain_weights=True)
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
