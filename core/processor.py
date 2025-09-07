import os
import cv2
import insightface
import core.globals
from core.config import get_face
from core.utils import rreplace

# 加载换脸模型
if os.path.isfile('inswapper_128.onnx'):
    face_swapper = insightface.model_zoo.get_model(
        'inswapper_128.onnx',
        providers=core.globals.providers
    )
else:
    quit('File "inswapper_128.onnx" does not exist!')


def process_video(source_img, frame_paths, face_index=0, from_right=False):
    source_face = get_face(cv2.imread(source_img))
    if source_face is None:
        raise RuntimeError("No face found in source image!")

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        face = get_face(frame, index=face_index, from_right=from_right)

        if face is None:
            print('S', end='', flush=True)  # S 表示 Skip（没检测到目标人脸）
            continue

        try:
            result = face_swapper.get(frame, face, source_face, paste_back=True)
            cv2.imwrite(frame_path, result)
            print('.', end='', flush=True)
        except Exception as e:
            print(f'Error swapping face on {frame_path}: {e}')
            raise


def process_img(source_img, target_path, output_file, face_index=0, from_right=False):
    frame = cv2.imread(target_path)
    face = get_face(frame, index=face_index, from_right=from_right)
    if face is None:
        raise RuntimeError("No face found in target image!")

    source_face = get_face(cv2.imread(source_img))
    if source_face is None:
        raise RuntimeError("No face found in source image!")

    try:
        result = face_swapper.get(frame, face, source_face, paste_back=True)
        cv2.imwrite(output_file, result)
        print("\n\nImage saved as:", output_file, "\n\n")
    except Exception as e:
        print(f'Error swapping face in image: {e}')
        raise
