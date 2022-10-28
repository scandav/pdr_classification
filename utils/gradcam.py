import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, GradCAMElementWise
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import torch
import uuid

class GradCamResnet():

    def __init__(self, model, save_parent_dir):

        self.save_dir = save_parent_dir / 'gradcam'
        self.save_dir.mkdir(exist_ok=True)

        # self.model = model
        target_layers = [model.module.layer4[-1]]
        self.cam = GradCAMElementWise(model=model, target_layers=target_layers, use_cuda=True)
        self.target = [ClassifierOutputTarget(0)]

    def run(self, input_tensor, fig_title, save_fname):
        
        fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.5))
        
        norm_img = input_tensor.cpu().numpy()[0] # first channel is grey, dataloader stacks them for resnet
        scaled_img = (norm_img - norm_img.min()) / (norm_img.max() - norm_img.min())
        bgr_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2BGR)

        grayscale_cam = self.cam(input_tensor=torch.unsqueeze(input_tensor, 0),
                            targets=self.target,
                            aug_smooth=True,
                            eigen_smooth=False)
        gradcam_img = show_cam_on_image(bgr_img, grayscale_cam[0, :], use_rgb=False, colormap=cv2.COLORMAP_RAINBOW)

        ax[0].imshow(bgr_img)
        ax[0].axis('off')

        ax[1].imshow(gradcam_img)
        ax[1].axis('off')

        fig.suptitle(fig_title)
        fig.tight_layout()
        fig.savefig(self.save_dir / save_fname)
        
        plt.close(fig)