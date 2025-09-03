import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
import time
from PIL import Image
from pathlib import Path

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import torchvision.transforms as T
import torchvision.models as models

# DEVICE = 'cuda'
# DEVICE = 'cpu'
DEVICE = 'mps'

# 统一resize到小尺寸，减少计算开销
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    h, w = img.shape[:2]
    if w >= h:
        new_wh = (768, 432)
    else:
        new_wh = (432, 768)
    img = cv2.resize(img, new_wh, interpolation=cv2.INTER_AREA)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def _torch_img_to_cv_uint8_rgb(img_torch):
    img_np = img_torch[0].permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return img_np

def estimate_global_affine(img1_rgb_uint8, img2_rgb_uint8):
    img1 = cv2.cvtColor(img1_rgb_uint8, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2_rgb_uint8, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(4000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
        M = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
        return M

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 6:
        M = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
        return M

    matches = sorted(matches, key=lambda m: m.distance)
    keep = max(30, int(len(matches) * 0.3))
    matches = matches[:keep]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    M, inliers = cv2.estimateAffinePartial2D(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000
    )
    if M is None:
        M = np.array([[1,0,0],[0,1,0]], dtype=np.float32)

    return M.astype(np.float32)

def affine_to_flow(M, H, W):
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    x2 = M[0,0]*xs + M[0,1]*ys + M[0,2]
    y2 = M[1,0]*xs + M[1,1]*ys + M[1,2]

    dx = x2 - xs
    dy = y2 - ys
    flow_bg = np.stack([dx, dy], axis=0)  # [2, H, W]
    return flow_bg  # numpy float32

def get_camera_motion(image1, image2):
    img1_np = _torch_img_to_cv_uint8_rgb(image1)
    img2_np = _torch_img_to_cv_uint8_rgb(image2)
    H, W = img1_np.shape[:2]

    M = estimate_global_affine(img1_np, img2_np)
    flow_bg_np = affine_to_flow(M, H, W).astype(np.float32)
    flow_bg = torch.from_numpy(flow_bg_np)[None, ...].to(DEVICE)

    return flow_bg

def get_human_mask(image_torch, threshold=0.5):
    if not hasattr(get_human_mask, "model"):
        model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(DEVICE)
        model.eval()
        get_human_mask.model = model
        get_human_mask.transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    model = get_human_mask.model
    transform = get_human_mask.transform

    x = image_torch / 255.0
    x = transform(x)

    with torch.no_grad():
        out = model(x)['out']  # [1,21,H,W]
        probs = torch.softmax(out, dim=1)
        # person类id=15
        person_mask = probs[:,15:16,:,:]  
        mask = (person_mask > threshold).float()

    return mask

save_counter = 0 

def images_to_gif(image_dir, output_path, duration=500):
    files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    images = [Image.open(os.path.join(image_dir, f)) for f in files]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

def viz(img, flo, args):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    global save_counter
    os.makedirs("optiflow_res", exist_ok=True)
    cv2.imwrite(f"optiflow_res/{save_counter}.png", img_flo[:, :, [2,1,0]].astype(np.uint8))
    save_counter += 1
    if args.remove_bg:
        images_to_gif("optiflow_res", "remove_bg.gif")
    else:
        images_to_gif("optiflow_res", "normal.gif")

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()

def extract_frames(VIDEO_PATH, OUTDIR, FRAME_INTERVAL) -> None:
    video_path = Path(VIDEO_PATH)
    outdir = Path(OUTDIR)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    outdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % FRAME_INTERVAL == 0:
            save_name = outdir / f"{frame_id:08d}.jpg"
            cv2.imwrite(str(save_name), frame)
        frame_id += 1

    cap.release()

def process_video_for_flow(video_path,
                           output_video_path,
                           model,
                           device,
                           frame_interval=1,
                           remove_bg=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if raw_w >= raw_h:
        vw, vh = 768, 432
    else:
        vw, vh = 432, 768
    out_w, out_h = vw * 2, vh
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))
    prev_image_tensor = None
    tmp_idx = 0
    with torch.no_grad():
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % frame_interval != 0:
                frame_id += 1
                continue
            tmp_path = f"__tmp_{tmp_idx}.jpg"
            cv2.imwrite(tmp_path, frame)      
            current_image_tensor = load_image(tmp_path).to(device)
            os.remove(tmp_path)
            tmp_idx += 1
            if prev_image_tensor is not None:
                padder = InputPadder(prev_image_tensor.shape)
                p1, p2 = padder.pad(prev_image_tensor, current_image_tensor)
                _, flow_up = model(p1, p2, iters=20, test_mode=True)
                flow_bg = get_camera_motion(prev_image_tensor, current_image_tensor)

                if remove_bg:
                    flow_up = flow_up - flow_bg

                mask = get_human_mask(p1)
                flow_up = flow_up * mask + (flow_bg) * (1-mask)

                prev_rgb = _torch_img_to_cv_uint8_rgb(prev_image_tensor)
                prev_bgr = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2BGR)
                flow_up_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
                flow_rgb = flow_viz.flow_to_image(flow_up_np)
                flow_bgr = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)
                concat = np.concatenate([prev_bgr, flow_bgr], axis=1)
                out.write(concat)
            prev_image_tensor = current_image_tensor
            frame_id += 1

    cap.release()
    out.release()

def demo(args):
    frames_dir = args.path.split(".")[0]
    extract_frames(args.path, frames_dir, 10)
    args.path = frames_dir

    model = torch.nn.DataParallel(RAFT(args))
    # model.load_state_dict(torch.load(args.model))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        vals = []
        vals_bg = []
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_bg = get_camera_motion(image1, image2)    
            if args.remove_bg:
                flow_up = flow_up-flow_bg # 减掉背景光流
            viz(image1, flow_up, args) # 光流可视化
            mask = get_human_mask(image1)
            flow_up = flow_up * mask
            h, w = flow_up.shape[-2:]
            human_ratio = mask.sum().item() / (h * w) # 人体占比
            # 全身平均
            # raw_mean = flow_up.norm(dim=1, keepdim=True).mean().item() # 
            # top20%平均
            raw_mean = torch.topk(flow_up.norm(dim=1).flatten(), max(1, int(0.2 * flow_up.numel() / 2))).values.mean().item()
            val = raw_mean / (max(human_ratio, 1e-6)**1.7) # 远小近大光流强度补偿
            vals.append(val)
            val_bg = flow_bg.norm(dim=1, keepdim=True).mean().item()
            vals_bg.append(val_bg)
            # print(f"[{(imfile1.split('/')[-1])} → {(imfile2.split('/')[-1])}] "
            #       f"当前帧人体区域平均光流强度: {val:.3f}")
        print("optiflow_human:", sum(vals)/len(vals))
        print("optiflow_bg:", sum(vals_bg)/len(vals_bg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove_bg', action='store_true', default=False) # 是否移除背景光流
    parser.add_argument('--model', help="restore checkpoint", default="/Users/wendell/Desktop/motion/filter/RAFT/models/raft-things.pth")
    parser.add_argument('--path', help="dataset for evaluation", default="demo_video.mp4")
    parser.add_argument('--small', action='store_true', help='use small model', default=False)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', default=False)
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)

    # model = torch.nn.DataParallel(RAFT(args))
    # model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))
    # model = model.module
    # model.to(DEVICE)
    # model.eval()
    # version = "static"
    # for idx, video in enumerate(os.listdir(version)):
    #     video_path = os.path.join(version, video)
    #     process_video_for_flow(video_path, f'{version}_{str(idx).zfill(3)}.mp4', model, DEVICE, remove_bg=args.remove_bg)
    # version = "dynamic"
    # for idx, video in enumerate(os.listdir(version)):
    #     video_path = os.path.join(version, video)
    #     process_video_for_flow(video_path, f'{version}_{str(idx).zfill(3)}.mp4', model, DEVICE, remove_bg=args.remove_bg)
