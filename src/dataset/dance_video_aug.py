import json
import random
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import cv2
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

def revert_rgb_to_alpha(alpha_rgb):
    img_matt_reverted = alpha_rgb[:, :, 0]
    img_matt_reverted = img_matt_reverted / 255.0
    return img_matt_reverted

def add_background(alpha_pred, image_ori, background_img):
    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
    background_img = cv2.resize(background_img, (image_ori.shape[1], image_ori.shape[0]))
    com_img = alpha_pred[..., None] * image_ori + (1 - alpha_pred[..., None]) * np.uint8(background_img)
    com_img = np.uint8(com_img)
    return com_img


def get_random_crop_params(width, height, scale, ratio, state):
    """Calculate parameters for a random sized crop using PyTorch's RNG."""
    torch.set_rng_state(state)
    area = height * width
    log_ratio = torch.log(torch.tensor(ratio))

    for attempt in range(3):
        target_area = torch.empty(1).uniform_(scale[0], scale[1]).item() * area
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round((target_area * aspect_ratio) ** 0.5))
        h = int(round((target_area / aspect_ratio) ** 0.5))

        if torch.rand(1).item() < 0.5:
            w, h = h, w

        if w <= width and h <= height:
            x = torch.randint(0, width - w + 1, (1,)).item()
            y = torch.randint(0, height - h + 1, (1,)).item()
            return x, y, w, h

    # Fallback to a smaller crop if necessary
    w = min(width, height)
    i = (width - w) // 2
    j = (height - w) // 2
    return i, j, w, w


def apply_crop_and_resize(frame, crop_params, target_size=(512, 512)):
    """Apply the calculated crop to a single frame and resize it."""
    cropped_frame = frame.crop((crop_params[0], crop_params[1], crop_params[0] + crop_params[2], crop_params[1] + crop_params[3]))
    resized_frame = cropped_frame.resize(target_size)
    return resized_frame


class HumanDanceVideoAugDatasetCrop(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        video_scale=(0.5, 1.0),
        video_ratio=(0.5, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fashion_meta.json"],
        background_dir="./backgrounds" 
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.video_scale = video_scale
        self.video_ratio = video_ratio
        self.available_backgrounds = [background_dir + '/'+ f for f in os.listdir(background_dir) if os.path.isfile(os.path.join(background_dir, f))]

        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform_vid = transforms.Compose(
            [
                # transforms.RandomResizedCrop(
                #     (height, width),
                #     scale=self.video_scale,
                #     ratio=self.video_ratio,
                #     interpolation=transforms.InterpolationMode.BILINEAR,
                # ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        self.pixel_transform_img = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(
                #     (height, width),
                #     scale=self.video_scale,
                #     ratio=self.video_ratio,
                #     interpolation=transforms.InterpolationMode.BILINEAR,
                # ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]
        matt_path = video_meta["matt_path"]

        try:
            video_reader = VideoReader(video_path)
            kps_reader = VideoReader(kps_path)
            if os.path.exists(matt_path):
                matt_reader = VideoReader(matt_path)
            else: 
                matt_reader = None

            assert len(video_reader) == len(
                kps_reader
            ), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"

            video_length = len(video_reader)

            clip_length = min(
                video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
            )
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(
                start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
            ).tolist()

            # read frames and kps
            vid_pil_image_list = []
            pose_pil_image_list = []
            background = np.array(Image.open(random.choice(self.available_backgrounds)))
            pose_dim_H, pose_dim_w = 0,0
            for index in batch_index:
                if matt_reader is not None:
                    og_img = video_reader[index].asnumpy()
                    H, W = og_img.shape[:2]
                    matt_img = matt_reader[index].asnumpy()
                    alpha = revert_rgb_to_alpha(matt_img)
                    comp_image = add_background(alpha, og_img, background)
                    vid_pil_image_list.append(Image.fromarray(comp_image))
                else:
                    og_img = video_reader[index]
                    H, W = og_img.shape[:2]
                    vid_pil_image_list.append(Image.fromarray(og_img.asnumpy()))
                img = kps_reader[index].asnumpy()
                # pose_dim_H, pose_dim_w = img.shape[:2]
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                pose_pil_image_list.append(Image.fromarray(img))
            
            state = torch.get_rng_state()
            
            if "HDTF" not in video_path:
                crop_params = get_random_crop_params(self.width, self.height, self.video_scale, self.video_ratio, state)
                for i in range(len(pose_pil_image_list)):
                    pose_pil_image_list[i] = apply_crop_and_resize(pose_pil_image_list[i], crop_params)
                    vid_pil_image_list[i] = apply_crop_and_resize(vid_pil_image_list[i], crop_params)
            
            
            ref_img_idx = random.randint(0, min(30, video_length - 1))
            if matt_reader is not None:
                og_img = video_reader[ref_img_idx].asnumpy()
                matt_img = matt_reader[ref_img_idx].asnumpy()
                alpha = revert_rgb_to_alpha(matt_img)
                ref_img = Image.fromarray(add_background(alpha, og_img, background))
                
            else:
                ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())

            # transform
            pixel_values_vid = self.augmentation(
                vid_pil_image_list, self.pixel_transform_vid, state
            )
            pixel_values_pose = self.augmentation(
                pose_pil_image_list, self.cond_transform, state
            )
            pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform_img, state)
            clip_ref_img = self.clip_image_processor(
                images=ref_img, return_tensors="pt"
            ).pixel_values[0]

            sample = dict(
                video_dir=video_path,
                pixel_values_vid=pixel_values_vid,
                pixel_values_pose=pixel_values_pose,
                pixel_values_ref_img=pixel_values_ref_img,
                clip_ref_img=clip_ref_img,
            )

            return sample
        
        except Exception as e:
            print(f"Error loading data at index {video_path}")

    def __len__(self):
        return len(self.vid_meta)
    
    
    
    
class HumanDanceVideoAugDatasetForLoRA(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        video_scale=(0.5, 1.0),
        video_ratio=(0.5, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fashion_meta.json"],
        background_dir="./backgrounds" 
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.video_scale = video_scale
        self.video_ratio = video_ratio
        self.available_backgrounds = [background_dir + '/'+ f for f in os.listdir(background_dir) if os.path.isfile(os.path.join(background_dir, f))]

        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform_vid = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.video_scale,
                    ratio=self.video_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        self.pixel_transform_img = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.video_scale,
                    ratio=self.video_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]
        matt_path = video_meta["matt_path"]

        try:
            video_reader = VideoReader(video_path)
            kps_reader = VideoReader(kps_path)
            if os.path.exists(matt_path):
                matt_reader = VideoReader(matt_path)
            else: 
                matt_reader = None

            assert len(video_reader) == len(
                kps_reader
            ), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"

            video_length = len(video_reader)

            clip_length = min(
                video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
            )
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(
                start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
            ).tolist()

            # read frames and kps
            vid_pil_image_list = []
            pose_pil_image_list = []
            background = np.array(Image.open(random.choice(self.available_backgrounds)))
            pose_dim_H, pose_dim_w = 0,0
            for index in batch_index:
                if matt_reader is not None:
                    og_img = video_reader[index].asnumpy()
                    H, W = og_img.shape[:2]
                    matt_img = matt_reader[index].asnumpy()
                    alpha = revert_rgb_to_alpha(matt_img)
                    comp_image = add_background(alpha, og_img, background)
                    vid_pil_image_list.append(Image.fromarray(comp_image))
                else:
                    og_img = video_reader[index]
                    H, W = og_img.shape[:2]
                    vid_pil_image_list.append(Image.fromarray(og_img.asnumpy()))
                img = kps_reader[index].asnumpy()
                # pose_dim_H, pose_dim_w = img.shape[:2]
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                pose_pil_image_list.append(Image.fromarray(img))
            
            state = torch.get_rng_state()
            
            
            ref_img_idx = random.randint(0, min(30, video_length - 1))
            if matt_reader is not None:
                og_img = video_reader[ref_img_idx].asnumpy()
                matt_img = matt_reader[ref_img_idx].asnumpy()
                alpha = revert_rgb_to_alpha(matt_img)
                ref_img = Image.fromarray(add_background(alpha, og_img, background))
                
            else:
                ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())

            # transform
            pixel_values_vid = self.augmentation(
                vid_pil_image_list, self.pixel_transform_vid, state
            )
            pixel_values_pose = self.augmentation(
                pose_pil_image_list, self.cond_transform, state
            )
            pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform_img, state)
            clip_ref_img = self.clip_image_processor(
                images=ref_img, return_tensors="pt"
            ).pixel_values[0]

            sample = dict(
                video_dir=video_path,
                pixel_values_vid=pixel_values_vid,
                pixel_values_pose=pixel_values_pose,
                pixel_values_ref_img=pixel_values_ref_img,
                clip_ref_img=clip_ref_img,
            )

            return sample
        
        except Exception as e:
            print(f"Error loading data at index {video_path}")

    def __len__(self):
        return len(self.vid_meta)



class HumanDanceVideoAugDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fashion_meta.json"],
        background_dir="./backgrounds" 
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.available_backgrounds = [background_dir + '/'+ f for f in os.listdir(background_dir) if os.path.isfile(os.path.join(background_dir, f))]

        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]
        matt_path = video_meta["matt_path"]
        
        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)
        if os.path.exists(matt_path):
            matt_reader = VideoReader(matt_path)
        else: 
            matt_reader = None

        assert len(video_reader) == len(
            kps_reader
        ), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"

        video_length = len(video_reader)

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # read frames and kps
        vid_pil_image_list = []
        pose_pil_image_list = []
        background = np.array(Image.open(random.choice(self.available_backgrounds)))
        for index in batch_index:
            if matt_reader is not None:
                og_img = video_reader[index].asnumpy()
                matt_img = matt_reader[index].asnumpy()
                alpha = revert_rgb_to_alpha(matt_img)
                comp_image = add_background(alpha, og_img, background)
                vid_pil_image_list.append(Image.fromarray(comp_image))
            else:
                og_img = video_reader[index]
                vid_pil_image_list.append(Image.fromarray(og_img.asnumpy()))
            img = kps_reader[index]
            pose_pil_image_list.append(Image.fromarray(img.asnumpy()))
       
                

        ref_img_idx = random.randint(0, video_length - 1)
        if matt_reader is not None:
            og_img = video_reader[ref_img_idx].asnumpy()
            matt_img = matt_reader[ref_img_idx].asnumpy()
            alpha = revert_rgb_to_alpha(matt_img)
            ref_img = Image.fromarray(add_background(alpha, og_img, background))
            
        else:
            ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())

        # transform
        state = torch.get_rng_state()
        pixel_values_vid = self.augmentation(
            vid_pil_image_list, self.pixel_transform, state
        )
        pixel_values_pose = self.augmentation(
            pose_pil_image_list, self.cond_transform, state
        )
        pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
        clip_ref_img = self.clip_image_processor(
            images=ref_img, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            video_dir=video_path,
            pixel_values_vid=pixel_values_vid,
            pixel_values_pose=pixel_values_pose,
            pixel_values_ref_img=pixel_values_ref_img,
            clip_ref_img=clip_ref_img,
        )

        return sample

    def __len__(self):
        return len(self.vid_meta)





