import torch
import torchvision.models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from .resnet_fpn_model import resnet50_fpn_backbone, resnet101_fpn_backbone
from .mask_rcnn import MaskRCNN


def create_model(num_classes, load_pretrain_weights=True):
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
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(backbone_name, pretrained=load_pretrain_weights)
    # The FPN backbone is only used for features, set it to eval mode to avoid issues with BatchNorm layers
    # backbone.body.eval()

    # Create the Mask R-CNN model
    model = torchvision.models.detection.MaskRCNN(backbone, num_classes=num_classes)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one for classification
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one for segmentation
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def build_custom_model(num_classes, use_resnet101=False, load_pretrain_weights=True):
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
        resnet_backbone = resnet50_fpn_backbone(norm_layer=FrozenBatchNorm2d, trainable_layers=5)

    # Create the Mask R-CNN model with the chosen backbone
    model = MaskRCNN(backbone=resnet_backbone, num_classes=num_classes)

    if load_pretrain_weights:
        # Pre-trained weights URL for COCO: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        weights_path = "./weights/maskrcnn_resnet50_fpn_coco.pth" if not use_resnet101 else "./weights/maskrcnn_resnet101_fpn_coco.pth"
        weights_dict = torch.load(weights_path, map_location="cpu")

        # Remove weights for box_predictor and mask_fcn_logits if they exist
        for k in list(weights_dict.keys()):
            if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                del weights_dict[k]

        print(model.load_state_dict(weights_dict, strict=False))
    
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
