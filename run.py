#!/usr/bin/env python3
import sys
import time
import shutil
import core.globals

# 检查 ffmpeg
if not shutil.which('ffmpeg'):
    print('ffmpeg is not installed. Read the docs: https://github.com/s0md3v/roop#installation.\n' * 10)
    quit()

# GPU 检查
if '--gpu' not in sys.argv:
    core.globals.providers = ['CPUExecutionProvider']
elif 'ROCMExecutionProvider' not in core.globals.providers:
    import torch
    if not torch.cuda.is_available():
        quit("You are using --gpu flag but CUDA isn't available or properly installed on your system.")

import glob
import argparse
import multiprocessing as mp
import os
from pathlib import Path
from core.processor import process_video, process_img
from core.utils import is_img, detect_fps, set_fps, create_video, add_audio, extract_frames
from core.config import get_face
import psutil
import cv2

pool = None
args = {}

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--face', required=True, help='use this face', dest='source_img')
parser.add_argument('-t', '--target', required=True, help='replace this face', dest='target_path')
parser.add_argument('-o', '--output', help='save output to this file', dest='output_file')
parser.add_argument('--keep-fps', help='maintain original fps', dest='keep_fps', action='store_true', default=False)
parser.add_argument('--gpu', help='use gpu', dest='gpu', action='store_true', default=False)
parser.add_argument('--keep-frames', help='keep frames directory', dest='keep_frames', action='store_true', default=False)
parser.add_argument('--cores', help='number of cores to use', dest='cores_count', type=int)
parser.add_argument("--face-index", type=int, default=0, help="选择第几个脸（默认=0，左数第一个）")
parser.add_argument("--from-right", action="store_true", help="是否从右边数")
args = vars(parser.parse_args())

if not args['cores_count']:
    args['cores_count'] = psutil.cpu_count() - 1

sep = "\\" if os.name == "nt" else "/"


def start_processing():
    start_time = time.time()
    frame_paths = args["frame_paths"]

    if args['gpu']:
        # 单进程，保持 GPU 调用
        process_video(args['source_img'], frame_paths, face_index=args['face_index'], from_right=args['from_right'])
    else:
        # 多进程，CPU 并行
        n = len(frame_paths) // args['cores_count']
        processes = []
        for i in range(0, len(frame_paths), n):
            p = pool.apply_async(process_video, args=(args['source_img'], frame_paths[i:i+n], args['face_index'], args['from_right']))
            processes.append(p)
        for p in processes:
            p.get()
        pool.close()
        pool.join()

    end_time = time.time()
    print(f"\nProcessing time: {end_time - start_time:.2f} seconds", flush=True)


def start():
    source_img = args['source_img']
    target_path = args['target_path']

    if not os.path.isfile(source_img):
        quit(f"Source image not found: {source_img}")
    if not os.path.isfile(target_path):
        quit(f"Target file not found: {target_path}")

    if not args['output_file']:
        target_name = os.path.basename(target_path)
        args['output_file'] = f"swapped-{target_name}"

    global pool
    pool = mp.Pool(args['cores_count'])

    # 检查 source face
    test_face = get_face(cv2.imread(source_img))
    if not test_face:
        quit("No face detected in source image!")

    if is_img(target_path):
        # 处理图片
        process_img(source_img, target_path, args['output_file'], face_index=args['face_index'], from_right=args['from_right'])
        print("Swap successful!")
        return

    # 处理视频
    video_name = os.path.splitext(os.path.basename(target_path))[0]
    output_dir = os.path.join(os.path.dirname(target_path), video_name)
    Path(output_dir).mkdir(exist_ok=True)

    print("Detecting video's FPS...")
    fps = detect_fps(target_path)

    # 降 FPS
    if not args['keep_fps'] and fps > 30:
        this_path = os.path.join(output_dir, f"{video_name}.mp4")
        set_fps(target_path, this_path, 30)
        target_path, fps = this_path, 30
    else:
        shutil.copy(target_path, output_dir)

    print("Extracting frames...")
    extract_frames(target_path, output_dir)

    args['frame_paths'] = tuple(sorted(
        glob.glob(os.path.join(output_dir, "*.png")),
        key=lambda x: int(os.path.basename(x).replace(".png", ""))
    ))

    print("Swapping in progress...")
    start_processing()

    print("Creating video...")
    create_video(video_name, fps, output_dir)

    print("Adding audio...")
    add_audio(output_dir, target_path, args['keep_frames'], args['output_file'])

    print(f"\n\nVideo saved as: {args['output_file']}\n\n")
    print("Swap successful!")


if __name__ == "__main__":
    start()
