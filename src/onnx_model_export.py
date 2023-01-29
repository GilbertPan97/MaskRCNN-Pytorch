import os
import torch
import onnx
import onnxruntime
from onnxsim import simplify
from PIL import Image
from torchvision import transforms
from backbone import resnet50_fpn_backbone
from network_files import MaskRCNN


# project root dir and src root dir
ROOT_DIR = os.path.abspath("../")
SRC_ROOT_DIR = os.path.abspath("./")

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
    output_label = ['out_boxs', 'out_labels', 'out_scores', 'out_masks']
    torch.onnx.export(pytorch_model,                # model being run
                      img,                          # model input (or a tuple for multiple inputs)
                      onnx_model_name,              # where to save the model (can be a file or file-like object)
                      export_params=True,           # store the trained parameter weights inside the model file
                      opset_version=11,             # the ONNX version to export the model to
                      do_constant_folding=True,     # whether to execute constant folding for optimization
                      input_names=input_label,      # the model's input names
                      output_names=output_label,    # the model's output names
                      dynamic_axes=None)              # variable length axes


def verify_onnx_model(onnx_model_name, img_file):
    # model is an in-memory ModelProto
    model = onnx.load(onnx_model_name)
    # print("the model is:\n{}".format(model))

    # check the model
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print("  the model is invalid: %s" % e)
        exit(1)
    else:
        print("  the model is valid")

    # verify onnx model inference
    ort_session = onnxruntime.InferenceSession(onnx_model_name)
    original_img = Image.open(img_file).convert('RGB')
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    input_img = torch.unsqueeze(img, dim=0).numpy()         # input_img: onnx model input image data
    ort_inputs = {'in_img': input_img}                       # define input dictionary
    try:
        ort_box = ort_session.run(['out_boxs'], ort_inputs)[0]     # onnx model inference
        ort_score = ort_session.run(['out_labels'], ort_inputs)[0]
        ort_mask = ort_session.run(['out_masks'], ort_inputs)[0]
        print("Onnx model inference result:\n{}".format(ort_mask))
    except:
        print("Onnx model inference fail.")
    # else:
    #     ort_output = np.squeeze(ort_output, 0)
    #     ort_output = np.clip(ort_output, 0, 255)
    #     ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
    #     cv2.imshow("MaskRCNN Detection Demo", ort_output)


def fix_onnx_model(onnx_save_path, export_dir):
    import onnx
    import onnx_graphsurgeon as gs

    gs_graph = gs.import_onnx(onnx.load(onnx_save_path))
    for i, node in enumerate(gs_graph.nodes):
        if "Reduce" in gs_graph.nodes[i].op and 'axes' not in node.attrs:
            # reduce all axes except batch axis
            gs_graph.nodes[i].attrs['axes'] = [i for i in range(1, len(gs_graph.nodes[i].inputs[0].shape))]

    new_onnx_graph = gs.export_onnx(gs_graph)
    patched_onnx_model_path = os.path.join(export_dir, 'patched.onnx')
    onnx.save(new_onnx_graph, patched_onnx_model_path)


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
    model = create_model(num_classes=num_classes)

    assert os.path.exists(WEIGHT_PATH), "{} file dose not exist.".format(WEIGHT_PATH)
    weights_dict = torch.load(WEIGHT_PATH, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # check onnx model dir
    if not os.path.exists(ONNX_MODEL_DIR):
        os.makedirs(ONNX_MODEL_DIR)

    # export onnx model
    onnx_model_path = os.path.join(ONNX_MODEL_DIR, "mask_rcnn.onnx")
    export_model_from_pytorch_to_onnx(model, device, IMG_PATH, onnx_model_path)     # export onnx model

    # simplify export onnx model
    onnx_sim_model_path = os.path.join(ONNX_MODEL_DIR, 'mask_rcnn_sim.onnx')
    onnx_model = onnx.load(onnx_model_path)
    onnx_sim_model, check = simplify(onnx_model)        # simplify onnx model
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_sim_model, onnx_sim_model_path)
    print('ONNX file simplified!')

    # check export onnx model
    verify_onnx_model(onnx_sim_model_path, IMG_PATH)    # verify onnx model

    fix_onnx_model(onnx_sim_model_path, ONNX_MODEL_DIR)


if __name__ == "__main__":
    main()