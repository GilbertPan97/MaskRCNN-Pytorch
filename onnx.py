import os
import torch
import onnx
import json
import argparse
import numpy as np
import onnxruntime
from onnxsim import simplify
from PIL import Image
from torchvision import transforms
from backbone import resnet50_fpn_backbone
from network_files import MaskRCNN

from utils.draw_box_utils import draw_objs


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


def export_model_from_pytorch_to_onnx(pytorch_model, device, img_file, onnx_model_name):
    # input of the model (from pil image to tensor, do not normalize image)
    original_img = Image.open(img_file).convert('RGB')      # load image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0).to(device)        # expand batch dimension to device

    pytorch_model.eval()
    # test model input
    im = torch.randn(1, 3, 256, 256)
    out = pytorch_model(im)
    print("out:", out)

    # export the model
    dy_axes = {'input': {0: 'batch', 2: 'height', 3: 'width'}}
    input_label = ['in_imgs']
    output_label = ['out_boxes', 'out_classes', 'out_scores', 'out_masks']
    torch.onnx.export(pytorch_model,                # model being run
                      img,                          # model input (or a tuple for multiple inputs)
                      onnx_model_name,              # where to save the model (can be a file or file-like object)
                      export_params=True,           # store the trained parameter weights inside the model file
                      opset_version=11,             # the ONNX version to export the model to
                      do_constant_folding=True,     # whether to execute constant folding for optimization
                      input_names=input_label,      # the model's input names
                      output_names=output_label,    # the model's output names
                      dynamic_axes=dy_axes)              # variable length axes


def verify_onnx_model(onnx_model_name, img_file, label_json_path):
    # model is an in-memory ModelProto
    model = onnx.load(onnx_model_name)

    # check the model
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print("  the model is invalid: %s" % e)
        exit(1)
    else:
        print("  the model is valid")

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # verify onnx model inference
    ort_session = onnxruntime.InferenceSession(onnx_model_name)
    original_img = Image.open(img_file).convert('RGB')
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    input_img = torch.unsqueeze(img, dim=0).numpy()         # input_img: onnx model input image data
    ort_inputs = {'in_imgs': input_img}                       # define input dictionary
    try:
        ort_boxes = ort_session.run(['out_boxes'], ort_inputs)[0]     # onnx model inference
        ort_classes = ort_session.run(['out_classes'], ort_inputs)[0]
        ort_scores = ort_session.run(['out_scores'], ort_inputs)[0]
        ort_masks = ort_session.run(['out_masks'], ort_inputs)[0]

        # squeeze: [channel, batch_size, height, width] -> [channel, height, width]
        ort_masks = np.squeeze(ort_masks, axis=1)

        plot_img = draw_objs(
            original_img, boxes=ort_boxes, classes=ort_classes,
            scores=ort_scores, masks=ort_masks, category_index=category_index,
            line_thickness=2, font='arial.ttf', font_size=100)
        plot_img.show("Inference results.")
    except:
        print("Onnx model inference fail.")


def fix_onnx_model(onnx_model_path, export_path):
    import onnx
    import onnx_graphsurgeon as gs

    gs_graph = gs.import_onnx(onnx.load(onnx_model_path))
    for i, node in enumerate(gs_graph.nodes):
        if "Reduce" in gs_graph.nodes[i].op and 'axes' not in node.attrs:
            # reduce all axes except batch axis
            gs_graph.nodes[i].attrs['axes'] = [i for i in range(1, len(gs_graph.nodes[i].inputs[0].shape))]

    new_onnx_graph = gs.export_onnx(gs_graph)
    onnx.save(new_onnx_graph, export_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Export a PyTorch Mask R-CNN model to ONNX, simplify it, and verify its correctness using a test image.")
    parser.add_argument('--weight', type=str, default="runs/exp/epoch_models/epoch_model.pth",
                        help='Transform pytorch model path')
    parser.add_argument('--export_folder', type=str, default="runs/exp/onnx_models",
                        help='Export ONNX model directory')
    parser.add_argument('--image', type=str, default="dataset/images/0211989685670_1711445097923_1711445097924_1711445098686.png",
                        help='Test model images path')
    parser.add_argument('--categories', type=str, default='cfg/categories.json',
                        help='Path to categories JSON file')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classes')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # get devices
    device = torch.device("cpu")    # export onnx model with cpu
    print("using {} device.".format(device))

    # create model & load train weights
    model = create_model(num_classes=args.num_classes)

    assert os.path.exists(args.weight), "{} file dose not exist.".format(args.weight)
    weights_dict = torch.load(args.weight, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # check onnx model dir
    if not os.path.exists(args.export_folder):
        os.makedirs(args.export_folder)

    # export onnx model
    onnx_model_path = os.path.join(args.export_folder, "mask_rcnn.onnx")
    export_model_from_pytorch_to_onnx(model, device, args.image, onnx_model_path)     # export onnx model

    # simplify export onnx model
    onnx_sim_model_path = os.path.join(args.export_folder, 'mask_rcnn_sim.onnx')
    onnx_model = onnx.load(onnx_model_path)
    onnx_sim_model, check = simplify(onnx_model)        # simplify onnx model
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_sim_model, onnx_sim_model_path)
    print('ONNX file simplified!')

    # fix onnx model nodes
    fixed_model_path = os.path.join(args.export_folder, "patched.onnx")
    fix_onnx_model(onnx_sim_model_path, fixed_model_path)

    # check export onnx model
    verify_onnx_model(onnx_model_path, args.image, args.categories)    # verify onnx model