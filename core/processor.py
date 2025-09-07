import os
import cv2
import insightface
import core.globals
from core.config import get_face
from core.utils import rreplace

if os.path.isfile('inswapper_128.onnx'):
    face_swapper = insightface.model_zoo.get_model('inswapper_128.onnx', providers=core.globals.providers)
else:
    quit('File "inswapper_128.onnx" does not exist!')


def process_video(source_img, frame_paths, face_index=0, from_right=False):
    """
    source_img: 源脸图片路径
    frame_paths: 目标视频帧路径列表
    face_index: 要替换的目标人脸索引
    from_right: 是否从右边数
    """
    source_face = get_face(cv2.imread(source_img), index=0)  # 源脸通常取第一个
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        try:
            face = get_face(frame, index=face_index, from_right=from_right)
            if face:
                result = face_swapper.get(frame, face, source_face, paste_back=True)
                cv2.imwrite(frame_path, result)
                print('.', end='', flush=True)
            else:
                print('S', end='', flush=True)
        except Exception as e:
            print('E', end='', flush=True)
            pass


def process_img(source_img, target_path, output_file, face_index=0, from_right=False):
    """
    处理单张图片的换脸
    """
    frame = cv2.imread(target_path)
    source_face = get_face(cv2.imread(source_img), index=0)  # 源脸默认取第一个
    face = get_face(frame, index=face_index, from_right=from_right)
    if face:
        result = face_swapper.get(frame, face, source_face, paste_back=True)
        cv2.imwrite(output_file, result)
        print("\n\nImage saved as:", output_file, "\n\n")
    else:
        print("\n\nNo face detected in target image.\n\n")
