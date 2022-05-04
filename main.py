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
        if len(tensor_img.shape) == 4:
            list_max_cls_scores = []
            for i in range(tensor_img.shape[0]):
                numpy_img = tensor_img[i].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

                from nanodet.data.transform.warp import ShapeTransform
                from nanodet.data.transform.color import color_aug_and_norm
                import functools

                meta = ShapeTransform(cfg.data.val.keep_ratio, **cfg.data.val.pipeline)({"img": numpy_img}, cfg.data.val.input_size)
                meta = functools.partial(color_aug_and_norm, kwargs=cfg.data.val.pipeline)(meta)

                processed_tensor_img = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device).type(torch.cuda.FloatTensor).unsqueeze(0)

                preds = self.model(processed_tensor_img)

                cls_scores = preds.split(
                    [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
                )[0]

                max_cls_scores = torch.max(cls_scores.sigmoid()[0], dim=0)[0]

                list_max_cls_scores.append(max_cls_scores)
            return torch.stack(list_max_cls_scores)
    
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

    # convert img to tensor
    input_   = torch.from_numpy(img).permute(2,0,1).to(device).type(torch.cuda.FloatTensor).unsqueeze(0)

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
