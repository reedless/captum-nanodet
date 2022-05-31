import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import cv2
from captum.attr import (Deconvolution, DeepLift, DeepLiftShap,
                         FeatureAblation, GradientShap, GuidedBackprop,
                         GuidedGradCam, InputXGradient, IntegratedGradients,
                         Occlusion, Saliency)
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from sklearn.metrics.pairwise import cosine_similarity


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
    
def average_cosine_similarity(image_path, model_path, config_path):
    device = torch.device('cuda')

    # set up
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # load image
    img = cv2.imread(image_path)

    # load config
    load_config(cfg, config_path)
    logger = Logger(-1, use_tensorboard=False)

    # load model
    wrapper = WrapperModel(cfg, model_path, logger, device=device)

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

    # convert to tensor, define baseline and baseline distribution
    input_ = torch.from_numpy(numpy_img_normalised.transpose(2, 0, 1)).to(device).type(torch.cuda.FloatTensor).unsqueeze(0)

    pred_class = 0

    # Integrated Gradients
    ig = IntegratedGradients(wrapper)
    ig_attributions, _ = ig.attribute(input_,
                                    target=pred_class,
                                    return_convergence_delta=True)

    # Saliency
    saliency = Saliency(wrapper)
    saliency_attributions = saliency.attribute(input_, target=pred_class)

    # InputXGradient
    inputxgradient = InputXGradient(wrapper)
    inputxgradient_attributions = inputxgradient.attribute(input_, target=pred_class)

    # Deconvolution
    deconv = Deconvolution(wrapper)
    deconv_attributions = deconv.attribute(input_, target=pred_class)

    # Guided Backprop
    gbp = GuidedBackprop(wrapper)
    gbp_attributions = gbp.attribute(input_, target=pred_class)

    attrs = [ig_attributions,
                saliency_attributions, inputxgradient_attributions, 
                deconv_attributions, gbp_attributions]
    processed_attrs = [process_attr(attr).reshape((-1)) for attr in attrs]
    sim_mat = cosine_similarity(np.stack(processed_attrs))

    average_cosine_similarity = (np.sum(sim_mat) - len(attrs))/ (len(attrs)**2 - len(attrs))
    print(average_cosine_similarity)
    return average_cosine_similarity


def process_attr(attributions):
    # C, H, W -> H, W, C
    attributions = attributions[0].permute(1,2,0).detach().cpu().numpy()

    # flattern to 1D
    attributions = np.sum(np.abs(attributions), axis=-1)

    # normalise attributions to [0,1]
    attributions -= np.min(attributions)
    attributions /= np.max(attributions)

    return attributions
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, default="dataset/night/20201201_000505.jpg")
    parser.add_argument("-w", "--weights_path", type=str, default='assets/nanodet-500/nanodet-500.pth')
    parser.add_argument("-c", "--config_path", type=str, default='assets/nanodet-500/nanodet-500.pth_train_config.yml')
    args = parser.parse_args()
    average_cosine_similarity(image_path=args.image_path, 
                              model_path=args.weights_path, 
                              config_path=args.config_path)
