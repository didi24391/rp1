import insightface
import core.globals

face_analyser = insightface.app.FaceAnalysis(
    name='buffalo_l',
    providers=core.globals.providers
)
face_analyser.prepare(ctx_id=0, det_size=(640, 640))


def get_face(img_data, index=0, from_right=False):
    """
    获取指定索引的人脸
    :param img_data: 输入图像
    :param index: 人脸索引 (0 表示第一个)
    :param from_right: 是否从右往左数
    """
    analysed = face_analyser.get(img_data)
    if not analysed:
        return None

    # 按照 bbox[0] 排序（从左到右）
    analysed_sorted = sorted(analysed, key=lambda x: x.bbox[0])

    # 从右数
    if from_right:
        analysed_sorted = list(reversed(analysed_sorted))

    # 取指定 index
    if index < len(analysed_sorted):
        return analysed_sorted[index]
    return None
