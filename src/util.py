import os
import cv2
from face_recognition import load_image_file, face_encodings
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def get_encodings(path, jitters=100):
    files = os.listdir(path)
    known_face_names = []
    known_face_encodings = []
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):

            name = file.split('.')[0]
            known_face_names.append(name)
            img = load_image_file(path+'/'+file)
            try:
                encodings = face_encodings(img, num_jitters=jitters)[0]
                known_face_encodings.append(encodings)
            except Exception as e:
                print(file, '无法识别')
        # print(known_face_encodings)
    return known_face_names, known_face_encodings


# def create_sql(path):
#     # 连接数据库
#     conn = pymysql.connect(
#         host='127.0.0.1',
#         user='jiage',
#         db='face',
#         passwd='wjc20160813',
#         port=3306,
#         #charset='utf-8'
#     )
#     cursor = conn.cursor()
#     # known_face_names, known_face_encodings = get_encodings(path)
#
#     with open('known_face_names', 'rb') as f:
#         known_face_names = pickle.load(f)
#     with open('known_face_encodings', 'rb') as f:
#         known_face_encodings = pickle.load(f)
#
#     for name, face_encoding in zip(known_face_names, known_face_encodings):
#
#         cursor.execute('INSERT INFO `face_encodings`(`name`,`face_encoding`) VALUES(`{0}`,`{1}`);'.format(name, ','.join([str(x) for x in face_encoding])))


def put_chinese(cvframe, pos, text, color, size):
    cv2img = cv2.cvtColor(cvframe, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)

    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印

    # 获取中文字体路径
    pdir = os.path.dirname(__file__)
    ppdir = os.path.dirname(pdir)

    font = ImageFont.truetype(ppdir+"/font/msjh.ttf", size, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
    draw.text(pos, text, color, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
     
    # PIL图片转cv2 图片
    frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return frame


if __name__ == "__main__":
    # create_sql('./photos/')
    get_encodings('./photos/')





