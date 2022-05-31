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

    # set up
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # load image
    image_path = 'dataset/night/20201201_000505.jpg'
    img = cv2.imread(image_path)
    epochs = [300, 500]

    for epoch in epochs:
        # file paths
        config_path = f'assets/nanodet-{epoch}/nanodet-{epoch}.pth_train_config.yml'
        model_path = f'assets/nanodet-{epoch}/nanodet-{epoch}.pth'

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
        attributions, delta = ig.attribute(input_,
                                        target=pred_class,
                                        return_convergence_delta=True)
        print('Integrated Gradients Convergence Delta:', delta)
        save_attr_mask(attributions, numpy_img_warped[:,:,::-1], 'IntegratedGradients', epoch)

        # Saliency
        saliency = Saliency(wrapper)
        attributions = saliency.attribute(input_, target=pred_class)
        save_attr_mask(attributions, numpy_img_warped[:,:,::-1], 'Saliency', epoch)

        # InputXGradient
        inputxgradient = InputXGradient(wrapper)
        attributions = inputxgradient.attribute(input_, target=pred_class)
        save_attr_mask(attributions, numpy_img_warped[:,:,::-1], 'InputXGradient', epoch)

        # Deconvolution
        deconv = Deconvolution(wrapper)
        attributions = deconv.attribute(input_, target=pred_class)
        save_attr_mask(attributions, numpy_img_warped[:,:,::-1], 'Deconvolution', epoch)

        # Guided Backprop
        gbp = GuidedBackprop(wrapper)
        attributions = gbp.attribute(input_, target=pred_class)
        save_attr_mask(attributions, numpy_img_warped[:,:,::-1], 'GuidedBackprop', epoch)


def save_attr_mask(attributions, img, algo_name, epoch):
    # save attributions
    os.makedirs(f'attributions/{epoch}/{algo_name}', exist_ok=True)
    torch.save(attributions, f'attributions/{epoch}/{algo_name}_attributions.pt')

    # C, H, W -> H, W, C
    attributions = attributions[0].permute(1,2,0).detach().cpu().numpy()

    # flattern to 1D
    attributions = np.sum(np.abs(attributions), axis=-1)

    # normalise attributions
    attributions -= np.min(attributions)
    attributions /= np.max(attributions)

    # plot masks
    _, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 8))
    axs[0, 0].set_title('Attribution mask')
    axs[0, 0].imshow(attributions, cmap=plt.cm.inferno)
    axs[0, 0].axis('off')
    axs[0, 1].set_title(f'Overlay {algo_name} on Input image ')
    axs[0, 1].imshow(attributions, cmap=plt.cm.inferno)
    axs[0, 1].imshow(img, alpha=0.5)
    axs[0, 1].axis('off')
    plt.tight_layout()

    # save masks
    os.makedirs(f'outputs/{epoch}/{algo_name}', exist_ok=True)
    plt.savefig(f'outputs/{epoch}/{algo_name}_mask.png', bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    main()