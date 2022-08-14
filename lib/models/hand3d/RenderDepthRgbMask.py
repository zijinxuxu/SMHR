import torch
import torch.nn as nn
from pytorch3d.renderer import DirectionalLights, Materials, SfMPerspectiveCameras, mesh
class RenderDepthRgbMask(nn.Module):
  def __init__(self, rasterizer, shader_rgb, shader_mask, cameras, multi=False):
    super(RenderDepthRgbMask, self).__init__()
    self.cameras = cameras
    self.rasterizer = rasterizer
    self.shader_rgb = shader_rgb
    self.shader_mask = shader_mask
    self.lights1 = DirectionalLights(
            ambient_color=((1, 1, 1),),
            diffuse_color=((0., 0., 0.),),
            specular_color=((0., 0., 0.),),
            direction=((0, 0, 1),),
            device='cuda',
        )
    self.lights2 = DirectionalLights(direction=((0, 0, 1),), device='cuda')
    self.lights = self.lights1
    self.materials = Materials(device='cuda')
    self.multi = multi

  def reset(self, device):
    if self.multi:
      cameras = self.cameras.clone().to(device)
      lights = self.lights.clone().to(device)
      materials = self.materials.clone().to(device)
    else:
      cameras = self.cameras
      lights = self.lights
      materials = self.materials
    return cameras, lights, materials

  def forward(self, meshes):
    cameras, lights, materials = self.reset(meshes.device)
    # check for cameras' numbers 4/3?
    fragments = self.rasterizer(meshes, cameras=cameras)
    image = self.shader_rgb(fragments, meshes, cameras=cameras, lights=lights, materials=materials)
    mask = self.shader_mask(fragments, meshes, cameras=self.cameras)
    return image, mask, fragments

  def render_rgb(self, meshes):
    cameras, lights, materials = self.reset(meshes.device)
    fragments = self.rasterizer(meshes, cameras=cameras)
    image = self.shader_rgb(fragments, meshes, cameras=cameras)
    return image

  def render_depth(self, meshes):
    cameras, lights, materials = self.reset(meshes.device)
    fragments = self.rasterizer(meshes, cameras=cameras)
    return fragments

  def render_mask(self, meshes):
    cameras, lights, materials = self.reset(meshes.device)
    fragments = self.rasterizer(meshes, cameras=cameras)
    mask = self.shader_mask(fragments, meshes, cameras=cameras)
    return mask