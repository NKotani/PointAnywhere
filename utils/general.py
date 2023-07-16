import os
from pathlib import Path
import cv2


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def name2cls(name:str):
    cls = 0
    with open('./utils/cocoName.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    for i, line in enumerate(lines):
        if name == line:
            cls = i
    return cls

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.general import colors'

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0), # 左下の点
          font_scale=4,
          font_thickness=16,
          text_color=(255, 255, 255), # white
          text_color_bg=(128, 128, 128) # background
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x-3, y-text_h-20), (x+text_w+3, y), text_color_bg, -1, cv2.LINE_AA) # LINE_AAで曲線に適したアンチエイリアスになる
    cv2.putText(img, text, (x-1, y-16), font, font_scale, text_color, font_thickness,cv2.LINE_AA) # 縦座標を広げないと小文字のyとかが下突き抜ける
    return text_size

def skip_sphere(name='score_great_mid.txt', sphere='R0000000'):
    already = False
    with open('./Result/'+name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines] # 2 0.5688492063492063 R0010066
    for line in enumerate(lines):
        # print(line[1])
        if sphere in line[1]:
            already = True
    return already
