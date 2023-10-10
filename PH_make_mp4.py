import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import SimpleITK as sitk
from PIL import Image
import imageio
from PIL import ImageDraw
import os


def make_mp4(mhd_file_path, output_folder):
    image = sitk.ReadImage(mhd_file_path)
    image_array = sitk.GetArrayFromImage(image)

    frames = []

    for threshold in range(257):
        image_array[image_array <= threshold] = 0
        frames.append(Image.fromarray(image_array))

    # 各フレームに閾値をテキストとして追加
    frames_with_text = []

    for i, frame in enumerate(frames):
        img_copy = frame.copy()
        draw = ImageDraw.Draw(img_copy)
        text = f"{i}"
        draw.text((3, 3), text, fill=255)  # 位置(10,10)にテキストを描画
        frames_with_text.append(img_copy)

    # 画像リストをMP4に変換
    mp4_with_text_output_path = os.path.join(
        output_folder, os.path.basename(mhd_file_path).split(".")[0] + ".mp4"
    )
    with imageio.get_writer(mp4_with_text_output_path, mode="I", fps=10) as writer:
        for frame in frames_with_text:
            writer.append_data(np.array(frame))


for curDir, dirs, files in os.walk("data/mhd"):
    # print(files)
    for file in files:
        if file.endswith(".mhd"):
            print(file)
            make_mp4(os.path.join(curDir, file), "result/animations/")
