import random
from turtle import pd
from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSION_LAYERS
import clip


class PromptLoRAAdapter(nn.Module):
    def __init__(self, channels=80, rank=8):
        super().__init__()

        self.lora_A = nn.Linear(channels, rank, bias=False)
        self.lora_B = nn.Linear(rank, channels, bias=False)
        
        nn.init.zeros_(self.lora_B.weight)
        
        self.scaling = 0.01

    def forward(self, x):
        lora_delta = self.lora_B(self.lora_A(x))
        return x + (lora_delta * self.scaling)
    

class WeatherPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.weather_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)        
        )
        
        self.time_head = nn.Linear(256, 2) # day/night
        self.weather_head = nn.Linear(256, 2) # rainy/sunny
        
        self.loss_scale = 0.7
        
        self.weather_prompt_map = {
            ('clear', 'day'): "Clear day, high visibility, reliable camera textures, sharp LiDAR geometry.",
            ('rainy', 'day'): "Rainy day, water spray, blurred camera visibility, scattered LiDAR noise.",
            ('clear', 'night'): "Clear night, low camera contrast, street light glare, reliable LiDAR depth.",
            ('rainy', 'night'): "Rainy night, heavy camera glare, reflections, noisy LiDAR returns, low visibility."
        }
      
    def forward(self, img_feats, img_metas, img_voxel_feats):
        wB, wV, wC, wH, wW = img_feats[0].shape
        w_feat = img_feats[0].view(wB * wV, wC, wH, wW)
        w_feat = self.gap(w_feat).view(wB * wV, -1)
        w_feat = self.weather_layers(w_feat)
       
        time_logit = self.time_head(w_feat).view(wB, wV, 2)
        weather_logit = self.weather_head(w_feat).view(wB, wV, 2)
        
        fused_time = time_logit.mean(dim=1)
        fused_weather = weather_logit.mean(dim=1)
    
        time_prediction = torch.argmax(fused_time, dim=-1)
        weather_prediction = torch.argmax(fused_weather, dim=-1)
        
        gt_prompt = img_metas[0]['weather_prompt']['prompt'].split(',')[0].lower()
        weather_gt = torch.tensor(0 if 'clear' in gt_prompt else 1).to(img_voxel_feats.device)
        time_gt = torch.tensor(0 if 'day' in gt_prompt else 1).to(img_voxel_feats.device)
        
        if self.training:
            # day = 0, night = 1        
            loss_time = self.criterion(fused_time, time_gt.unsqueeze(0).long())
            
            # clear = 0, rainy = 1
            loss_weather = self.criterion(fused_weather, weather_gt.unsqueeze(0).long())
            
            loss = self.loss_scale * (loss_time + loss_weather)
            
        time = 'day' if time_prediction.item() == 0 else 'night'
        weather = 'clear' if weather_prediction.item() == 0 else 'rainy'
        prompt = self.weather_prompt_map.get((weather, time))
        
        if self.training:
            return prompt, loss
        else:
            return prompt, None


@FUSION_LAYERS.register_module()
class WeatherFuser(nn.Module):
    def __init__(self, feature_dim=80, clip_dim=512,
                 num_heads=4, use_lora=False, lora_rank=8, 
                 lora_alpha=16, lora_dropout=0.0) -> None:
        super().__init__()
        
        self.use_lora = use_lora

        clip_model, _ = clip.load("ViT-B/32", device='cuda')
        self.clip_model = clip_model.eval()
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        self.text_proj = nn.Linear(clip_dim, feature_dim)
        
        self.weather_pred = WeatherPrediction()
        
        self.lora = PromptLoRAAdapter(clip_dim, rank=lora_rank) if use_lora else None
        
        self.gate_generator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.Sigmoid()
        )
        
        self.weather_to_scalar = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.time_to_scalar = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
                
    def forward(self, img_voxel_feats, pts_voxel_feats, img_metas=None, img_feats=None, **kwargs):
        target_dtype = img_voxel_feats.dtype
        
        prompt, wloss = self.weather_pred(img_feats, img_metas, img_voxel_feats)
        
        weather_tokens = torch.stack([clip.tokenize(meta['weather_prompt']['prompt']).squeeze() for meta in img_metas]).to(img_voxel_feats.device)

        with torch.no_grad():
            weather_features = self.clip_model.encode_text(weather_tokens).to(target_dtype)
        
        if self.use_lora:
            weather_features = self.lora(weather_features)
          
        weather_features = self.text_proj(weather_features).unsqueeze(1)          
                
        B, C, D, H, W = img_voxel_feats.shape
        
        gates = self.gate_generator(weather_features.view(B, C))
        
        cam_gate, lidar_gate = torch.split(gates, C, dim=1)
        
        cam_gate = cam_gate.view(B, C, 1, 1, 1)
        lidar_gate = lidar_gate.view(B, C, 1, 1, 1)
        
        gated_cam = img_voxel_feats * cam_gate
        gated_lidar = pts_voxel_feats * lidar_gate
                
        alpha_w = self.weather_to_scalar(weather_features) 
        beta_t = self.time_to_scalar(weather_features)      
        
        cam_ratio = alpha_w * beta_t 
        lidar_ratio = 1.0 - cam_ratio
        
        fused_voxels = (gated_cam * cam_ratio.view(-1, 1, 1, 1, 1)) + \
                       (gated_lidar * lidar_ratio.view(-1, 1, 1, 1, 1))
        
        return fused_voxels, wloss