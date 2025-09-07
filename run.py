#!/usr/bin/env python3
import sys
import time
import shutil
import core.globals
import glob
import argparse
import multiprocessing as mp
import os
from pathlib import Path
from core.processor import process_video, process_img
from core.utils import is_img, detect_fps, set_fps, create_video, add_audio, extract_frames, rreplace
import psutil

if not shutil.which('ffmpeg'):
    print('ffmpeg is not installed. Read the docs: https://github.com/s0md3v/roop#installation.\n' * 10)
    quit()

if '--gpu' not in sys.argv:
    core.globals.providers = ['CPUExecutionProvider']
elif 'ROCMExecutionProvider' not in core.globals.providers:
    import torch
    if not torch.cuda.is_available():
        quit("You are using --gpu flag but CUDA isn't available or properly installed on your system.")

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--face', help='use this face', dest='source_img', required=True)
parser.add_argument('-t', '--target', help='replace this face', dest='target_path', required=True)
parser.add_argument('-o', '--output', help='save output to this file', dest='output_file')
parser.add_argument('--keep-fps', help='maintain original fps', dest='keep_fps', action='store_true', default=False)
parser.add_argument('--gpu', help='use gpu', dest='gpu', action='store_true', default=False)
parser.add_argument('--keep-frames', help='keep frames directory', dest='keep_frames', action='store_true', default=False)
parser.add_argument('--cores', help='number of cores to use', dest='cores_count', type=int)
parser.add_argument("--face-index", type=int, default=0, help="选择第几个脸（默认=0，左数第一个）")
parser.add_argument("--from-right", action="store_true", help="是否从右边数")

args = vars(parser.parse_args())

if not args['cores_count']:
    args['cores_count'] = psutil.cpu_count()-1

sep = "/" if os.name != "nt" else "\\"

pool = None

def start_processing():
    start_time = time.time()
    if args['gpu']:
        process_video(
            args['source_img'],
            args["frame_paths"],
            face_index=args['face_index'],
            from_right=args['from_right']
        )
        end_time = time.time()
        print(flush=True)
        print(f"Processing time: {end_time - start_time:.2f} seconds", flush=True)
        return

    frame_paths = args["frame_paths"]
    n = len(frame_paths)//(args['cores_count'])
    processes = []

    for i in range(0, len(frame_paths), n):
        p = pool.apply_async(
            process_video,
            args=(
                args['source_img'],
                frame_paths[i:i+n],
                args['face_index'],
                args['from_right']
            )
        )
        processes.append(p)

    for p in processes:
        p.get()
    pool.close()
    pool.join()
    end_time = time.time()
    print(flush=True)
    print(f"Processing time: {end_time - start_time:.2f} seconds", flush=True)

def main():
    target_path = args['target_path']
    if not os.path.isfile(args['source_img']):
        print("\n[WARNING] Please select an image containing a face.")
        return
    if not os.path.isfile(target_path):
        print("\n[WARNING] Please select a video/image to swap face in.")
        return

    if not args['output_file']:
        args['output_file'] = rreplace(target_path, "/", "/swapped-", 1) if "/" in target_path else "swapped-"+os.path.basename(target_path)

    global pool
    pool = mp.Pool(args['cores_count'])

    if is_img(target_path):
        process_img(
            args['source_img'],
            target_path,
            args['output_file'],
            face_index=args['face_index'],
            from_right=args['from_right']
        )
        print("Image swap successful!")
        return

    video_name = os.path.basename(target_path)
    video_name = os.path.splitext(video_name)[0]
    output_dir = os.path.join(os.path.dirname(target_path), video_name)
    Path(output_dir).mkdir(exist_ok=True)

    print("Detecting video's FPS...")
    fps = detect_fps(target_path)
    if not args['keep_fps'] and fps > 30:
        this_path = os.path.join(output_dir, video_name + ".mp4")
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

    save_path = args['output_file'] if args['output_file'] else os.path.join(output_dir, video_name + ".mp4")
    print("\nVideo saved as:", save_path, "\nSwap successful!")

if __name__ == "__main__":
    main()
