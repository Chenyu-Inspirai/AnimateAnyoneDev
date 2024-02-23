import os
import random
import cv2
from tqdm import tqdm
from datetime import datetime

import numpy as np
import torch
import json
import copy
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_from_pil
from src.utils.lora_handler import LoraHandler
from tools.openpose_rescaler import draw, rescale_skeleton


class AnimateInferPipeline:
    def __init__(
        self,
        base_model_path,
        reference_unet_path,
        denoising_unet_path, 
        vae_path,
        pose_guider_path,
        motion_module_path,
        image_encoder_path,
        infer_config_path,
        lora_path=None,
        device='cuda',
        weight_dtype=torch.float16,
    ):
        self.weight_dtype = weight_dtype
        self.vae = AutoencoderKL.from_pretrained(vae_path,).to(device, dtype=self.weight_dtype)
        self.reference_unet = UNet2DConditionModel.from_pretrained(
                base_model_path,
                subfolder="unet",
            ).to(dtype=self.weight_dtype, device=device)
        
        infer_config = OmegaConf.load(infer_config_path)
        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            base_model_path,
            motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=self.weight_dtype, device=device)
        
        self.pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
                dtype=self.weight_dtype, device=device
        )
        
        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(
                image_encoder_path
            ).to(dtype=self.weight_dtype, device=device)
        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        self.scheduler = DDIMScheduler(**sched_kwargs)
            
        # load pretrained weights
        self.denoising_unet.load_state_dict(
            torch.load(denoising_unet_path, map_location="cpu"),
            strict=False,
        )
        self.reference_unet.load_state_dict(
            torch.load(reference_unet_path, map_location="cpu"),
        )
        self.pose_guider.load_state_dict(
            torch.load(pose_guider_path, map_location="cpu"),
        )
        
        pipe = Pose2VideoPipeline(
                vae=self.vae,
                image_encoder=self.image_enc,
                reference_unet=self.reference_unet,
                denoising_unet=self.denoising_unet,
                pose_guider=self.pose_guider,
                scheduler=self.scheduler,
            )
        
        # If we are using lora
        
        if lora_path is not None:
            lora_manager = LoraHandler(
            use_unet_lora=True, 
            unet_replace_modules=["TemporalTransformerBlock"]
            )
            
            _, _ = lora_manager.add_lora_to_model(
                use_lora=True, 
                model=pipe.denoising_unet, 
                replace_modules=lora_manager.unet_replace_modules, 
                dropout=0.0, 
                lora_path=lora_path, 
                r=128, 
                scale=1)
            
            print("Loaded temporal lora")
            
        # if lora_path is not None:
        #     pipe.load_lora_weights(lora_path, lora_scale=1)
        #     print("Loaded temporal lora")
            
        pipe = pipe.to(device, dtype=self.weight_dtype)
        self.pipeline = pipe
        
    
    def animate(self,
                ref_image, 
                pose_list, 
                width, 
                height,
                video_length,
                num_inference_steps,
                cfg,
                seed,
                context_frames=16):
        
        generator = torch.manual_seed(seed)
        if isinstance(ref_image, np.ndarray):
            ref_image = Image.fromarray(ref_image)
            
        video = self.pipeline(
            ref_image,
            pose_list,
            width=width,
            height=height,
            video_length=video_length,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg,
            generator=generator,
            context_frames=context_frames
        ).videos
        
        return video
    

def use_raw_vid(
    raw_video_dir,
    W,
    H,
    detector,
    saving_name,
    ref_pose_keypoints=None,
    start_frame = 0,
    end_frame = -1
):
    raw_vid = read_frames(raw_video_dir)[start_frame: end_frame]
    print("total_frames: ", len(raw_vid))
    driving_pose = []
    vid_key_points = []
    for frame in tqdm(raw_vid):
        frame = cv2.resize(np.array(frame), (W, H), interpolation=cv2.INTER_LINEAR)
        pose, score, keypoints = detector(frame, output_type='key_points')
        driving_pose.append(pose)
        vid_key_points.append(keypoints)

    filename = '/cephfs/SZ-AI/usr/liuchenyu/HaiLook/Moore-AnimateAnyone/assets/key_points/' + saving_name + '.json'
    
    save_keypoints_copy = copy.deepcopy(vid_key_points)
    for frame_data in save_keypoints_copy:
        for k, v in frame_data.items():
            if isinstance(v, dict):
                for key,val in v.items():
                    v[key] = val.tolist()
            else:
                frame_data[k] = v.tolist()
    with open(filename, 'w') as f:
        json.dump(save_keypoints_copy, f, indent=4)
    print("---------------Saved original keypoints-------------------")
    
    processed_pose_vid = []
    if ref_pose_keypoints is not None:
        targ = ref_pose_keypoints
        for frame in vid_key_points:
            new_frame = rescale_skeleton(frame, targ)
            new_frame_pil = draw(new_frame, W, H)
            processed_pose_vid.append(new_frame_pil)
        
        save_vid_filename = '/cephfs/SZ-AI/usr/liuchenyu/HaiLook/Moore-AnimateAnyone/assets/pose_vid/' + saving_name +'.mp4'
        save_videos_from_pil(processed_pose_vid, save_vid_filename, fps=30)
        print("---------------Saved rescaled dwpose video----------------")
    
    else:
        processed_pose_vid = driving_pose
        save_vid_filename = '/cephfs/SZ-AI/usr/liuchenyu/HaiLook/Moore-AnimateAnyone/assets/pose_vid/' + saving_name +'.mp4'
        save_videos_from_pil(processed_pose_vid, save_vid_filename, fps=30)
        print("---------------Saved dwpose video-------------------------")
    
    return processed_pose_vid, save_vid_filename



def use_vid_key_points(
    pose_keypoints_file,
    W,
    H,
    saving_name,
    ref_pose_keypoints=None
):
    with open(pose_keypoints_file, 'r') as file:
        vid_key_points = json.load(file)
        
    for frame_data in vid_key_points:
        for k, v in frame_data.items():
            if isinstance(v, dict):
                for key,val in v.items():
                    v[key] = np.array(val)
            else:
                frame_data[k] = np.array(v)
    
    processed_pose_vid = []
    save_vid_filename = '/cephfs/SZ-AI/usr/liuchenyu/HaiLook/Moore-AnimateAnyone/assets/pose_vid/' + saving_name +'.mp4'
    if ref_pose_keypoints is not None:
        targ = ref_pose_keypoints
        for frame in vid_key_points:
            new_frame = rescale_skeleton(frame, targ)
            processed_pose_vid.append(draw(new_frame, H, W))
        
        save_videos_from_pil(processed_pose_vid, save_vid_filename, fps=30)
        print("---------------Saved rescaled dwpose video----------------")
    
    else:
        for frame in vid_key_points:
            processed_pose_vid.append(draw(frame, H, W))
        save_videos_from_pil(processed_pose_vid, save_vid_filename, fps=30)
        print("---------------Saved dwpose video-------------------------")
            
    return processed_pose_vid, save_vid_filename