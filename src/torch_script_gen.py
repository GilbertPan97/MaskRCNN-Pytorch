import os
import torch
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
    traced_script_module = torch.jit.script(pytorch_model, img)
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
    model = create_model(num_classes=num_classes)

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

    # export onnx model
    model_save_path = os.path.join(MODELS_DIR, "traced_model.pt")
    generate_torch_script(model, device, IMG_PATH, model_save_path)     # export onnx model
    model_script = torch.jit.load(model_save_path)
    model_script.eval()
    with torch.no_grad():
        img_tensor_list = list()
        img_tensor_list.append(img)
        predictions = model_script([img.to(device)])[1][0]

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        predict_mask = predictions["masks"].to("cpu").numpy()


if __name__ == "__main__":
    main()