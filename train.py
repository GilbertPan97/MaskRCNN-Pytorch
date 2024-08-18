import os
import datetime
import argparse
import yaml

import torch

from utils import transforms
from utils.my_dataset_coco import CocoDetection
from utils import train_eval_utils as utils

from network_files.model import create_model, build_model, build_custom_model


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(hyp, opt, device):
    # save train info file to det_results_file & seg_results_file
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(opt.runs_dir):
        os.makedirs(opt.runs_dir)
    det_results_file = opt.runs_dir+f"/det_results{now}.txt"
    seg_results_file = opt.runs_dir+f"/seg_results{now}.txt"

    # define model input data
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # load train dataset
    train_dataset = CocoDetection(opt.dataset, "train", data_transform["train"])
    print('Using %g dataloader workers' % hyp['NUM_WORKERS'])

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=hyp['BatchSize'],
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=hyp['NUM_WORKERS'],
                                                    collate_fn=train_dataset.collate_fn)

    # load validation data set
    val_dataset = CocoDetection(opt.dataset, "val", data_transform["val"])
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=hyp['BatchSize'],
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=hyp['NUM_WORKERS'],
                                                  collate_fn=train_dataset.collate_fn)

    # create model with backbone: resnet50 or resnet101
    # if opt.backbone == 'resnet101':
    #     model = build_custom_model(num_classes=hyp['NUM_CLASSES'],
    #                                load_pretrain_weights=True,
    #                                use_resnet101=True)
    # else:
    #     model = build_custom_model(num_classes=hyp['NUM_CLASSES'],
    #                                load_pretrain_weights=True,
    #                                use_resnet101=False)
    if opt.backbone == 'resnet101':
        model = build_model(num_classes=hyp['NUM_CLASSES'],
                            load_pretrain_weights=True,
                            use_resnet101=True)
    else:
        model = build_model(num_classes=hyp['NUM_CLASSES'],
                            load_pretrain_weights=True,
                            use_resnet101=False)
    # model = create_model(num_classes=len(categorys),
    #                      load_pretrain_weights=True)
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=hyp['LEARNING_RATE'],
                                momentum=hyp['LEARNING_MOMENTUM'],
                                weight_decay=hyp['WEIGHT_DECAY'])

    # learning rate scheduler
    milestones = [int(hyp['EPOCHS'] * 1 / 3), int(hyp['EPOCHS'] * 2 / 3)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1)

    train_loss = []
    learning_rate = []
    val_map = []
    scaler = torch.cuda.amp.GradScaler()

    # Check if resuming from checkpoint
    if opt.resume:
        # Load checkpoint from specified path
        checkpoint = torch.load(opt.resume, map_location=device)

        # Load model state
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            raise KeyError("'model' key missing in the checkpoint.")
        
        # Load optimizer state
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            raise KeyError("'optimizer' key missing in the checkpoint.")
        
            # Load learning rate scheduler state
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            raise KeyError("'lr_scheduler' key missing in the checkpoint.")

        # Load scaler state if it exists
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        else:
            print("'scaler' key missing in the checkpoint, but this is fine if mixed precision is not used.")
    else:
        print("Not resuming from checkpoint. Using default weights and starting fresh.")

    for epoch in range(0, hyp['EPOCHS']):
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
        if opt.runs_dir:
            epoch_dir = os.path.join(opt.runs_dir, 'epoch_models')
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)
            
            epoch_paths = [os.path.join(epoch_dir, 'epoch_model.pth')]
            if (epoch + 1) % 50 == 0:
                epoch_paths.append(os.path.join(epoch_dir, f'epoch_model{epoch:04}.pth'))
            
            for epoch_path in epoch_paths:
                save_files = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }
                torch.save(save_files, epoch_path)

    # plot loss and lr (learn rate) curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from utils.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate, opt.runs_dir)

    # plot mAP (Mean Average Precision) curve
    if len(val_map) != 0:
        from utils.plot_curve import plot_map
        plot_map(val_map, opt.runs_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse arguments for model training')
    
    parser.add_argument('--dataset', type=str, default='dataset', 
                        help='COCO dataset root directory containing images and annotations')
    
    parser.add_argument('--cfg', type=str, default='cfg/train.yaml', 
                        help='Path to the YAML config file with model and training parameters')
    
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet101',
                        help='Backbone network to use: resnet50 or resnet101')
    
    parser.add_argument('--runs_dir', type=str, default='runs/exp/', 
                        help='Directory to save training records, logs, and checkpoints')

    parser.add_argument('--resume', type=str, default='', 
                        help='Directory to resume training from a checkpoint or directory containing pretrained weights')
    
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_arguments()
    cfg = load_config(opt.cfg)
    print("Loaded config from: ", opt.cfg)
    print("Train config is:\n", cfg)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {} device training.".format(device.type))
    train(cfg, opt, device)
