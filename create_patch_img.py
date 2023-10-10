import os

import numpy as np
import SimpleITK as sitk
from PIL import Image


def create_patch_img(mhd_file_name, difine_point):
    name_lists = ["inf", "sr", "hr"]
    patch_lists = []
    patch_all_img = np.zeros((64, 64 * 3))

    for name in name_lists:
        mhd_file_path = os.path.join("data/", mhd_file_name + f"_{name}.mhd")

        image = sitk.ReadImage(mhd_file_path)
        image_array = sitk.GetArrayFromImage(image)
        image_patch = image_array[
            difine_point : difine_point + 64, difine_point : difine_point + 64
        ]

        patch_lists.append(image_patch)

    patch_all_img = np.hstack((patch_lists[0], patch_lists[1], patch_lists[2]))
    patch_all_img_sitk = sitk.GetImageFromArray(patch_all_img)

    output_file_path = os.path.join(
        "data/mhd", mhd_file_name + f"_all_{difine_point}.mhd"
    )
    sitk.WriteImage(patch_all_img_sitk, output_file_path)
    # save image as png
    patch_all_img = patch_all_img.astype(np.uint8)
    patch_all_img = Image.fromarray(patch_all_img)
    patch_all_img.save(
        "data/png/" + mhd_file_name + f"_all_{difine_point}.png"
    )


for i in range(370, 1200, 64):
    create_patch_img("0_5", i)
