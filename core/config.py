import insightface
import core.globals

face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=core.globals.providers)
face_analyser.prepare(ctx_id=0, det_size=(640, 640))


def get_face(img_data, index=0, from_right=False):
    """
    检测并返回指定人脸
    :param img_data: 输入图像
    :param index: 第几个（默认0，表示第一个）
    :param from_right: 是否从右边开始计数（默认False=从左数）
    """
    analysed = face_analyser.get(img_data)
    try:
        faces = sorted(analysed, key=lambda x: x.bbox[0])  # 按 x 坐标排序，从左到右
        if from_right:
            return faces[-(index + 1)]  # 从右数
        else:
            return faces[index]         # 从左数
    except IndexError:
        return None
