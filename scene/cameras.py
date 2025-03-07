import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from PIL import Image
import os
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", K=None, 
                 sky_mask=None, normal=None, depth=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.sky_mask = sky_mask
        self.normal = normal
        self.depth = depth

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
    

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:

                # mask path
                default_mask_path = ".\\data\\peoplecarstreeview\\mask\\" + self.image_name + ".npy"

              
                if os.path.exists(default_mask_path):
                    
                    default_mask = np.load(default_mask_path).astype(np.float32)
                   
                    default_mask_tensor = torch.tensor(default_mask, device=self.data_device)

                    if default_mask_tensor.shape != self.original_image.shape[1:]:
                        default_mask_tensor = torch.nn.functional.interpolate(
                            default_mask_tensor.unsqueeze(0).unsqueeze(0), 
                            size=(self.image_height, self.image_width),
                            mode='nearest'
                        ).squeeze(0).squeeze(0)

                    self.sky_mask = default_mask_tensor
                else:
                 
                    print(f"Warning: The default mask file at {default_mask_path} does not exist!")
                
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)


        self.K = torch.tensor([[K[0], 0, K[2]],
                               [0, K[1], K[3]],
                               [0, 0, 1]]).to(self.data_device).to(torch.float32)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

