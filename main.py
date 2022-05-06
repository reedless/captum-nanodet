import os

import numpy as np
import torch

import cv2
from captum.attr import (Deconvolution, DeepLift, DeepLiftShap,
                         FeatureAblation, GradientShap, GuidedBackprop,
                         GuidedGradCam, InputXGradient, IntegratedGradients,
                         Occlusion, Saliency)
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight


class WrapperModel(torch.nn.Module):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        super().__init__()

        self.cfg = cfg
        self.device = device
        
        self.num_classes = cfg['model']['arch']['head']['num_classes']
        self.reg_max = cfg['model']['arch']['head']['reg_max']
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, _: storage)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval()

    def forward(self, tensor_img):
        tensor_img = tensor_img.to(self.device)

        if len(tensor_img.shape) == 3:
            tensor_img = tensor_img.unsqueeze(0)

        if len(tensor_img.shape) == 4:
            preds = self.model(tensor_img)
            cls_scores = preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )[0]
            
            max_cls_scores = torch.max(cls_scores.sigmoid(), dim=1)[0]
            return max_cls_scores
        else:
            raise ValueError('tensor_img.shape must be (N, C, H, W) or (C, H, W)')
    
def main():
    device = torch.device('cuda')

    # file paths
    config_path = 'demo/nanodet-plus-m_416.yml'
    model_path = 'demo/nanodet-plus-m_416_checkpoint.ckpt'
    image_path = 'demo/000252.jpg'

    # set up
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # load config
    load_config(cfg, config_path)
    logger = Logger(-1, use_tensorboard=False)

    # load model
    wrapper = WrapperModel(cfg, model_path, logger, device=device)

    # load image
    img = cv2.imread(image_path)

    # preprocessing
    raw_height = img.shape[0]
    raw_width  = img.shape[1]
    dst_width, dst_height = cfg.data.val.input_size
    ResizeM = np.eye(3)
    ResizeM[0, 0] *= dst_width / raw_width
    ResizeM[1, 1] *= dst_height / raw_height

    # scaling only
    numpy_img_warped = cv2.warpPerspective(img, 
                                        ResizeM, 
                                        dsize=tuple(cfg.data.val.input_size), 
                                        flags = cv2.INTER_LINEAR, 
                                        borderMode = cv2.BORDER_CONSTANT)

    # normalise
    mean, std = cfg.data.val.pipeline["normalize"]
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    numpy_img_normalised = ((numpy_img_warped.astype(np.float32) / 255) - mean) / std

    # numpy to pytorch
    input_ = torch.from_numpy(numpy_img_normalised.transpose(2, 0, 1)).to(device).type(torch.cuda.FloatTensor).unsqueeze(0)

    # run model
    thres = 0.35
    class_scores = wrapper(input_)
    pred_classes = [i for i, x in enumerate(class_scores[0] > thres) if x]

    for pred_class in pred_classes:
        # Integrated Gradients
        ig = IntegratedGradients(wrapper)
        attributions, delta = ig.attribute(input_,
                                        target=pred_class,
                                        return_convergence_delta=True)
        print('Integrated Gradients Convergence Delta:', delta)
        print(attributions)

if __name__ == "__main__":
    main()
