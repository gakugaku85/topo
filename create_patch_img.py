import os

import numpy as np
import SimpleITK as sitk
from PIL import Image


def create_patch_img(mhd_file_name):
    name_lists = ["inf", "sr", "hr"]
    image_lists = []
    patch_lists = []
    patch_all_img = np.zeros((64, 64 * 3))

    os.makedirs(f"data/{mhd_file_name}", exist_ok=True)

    for name in name_lists:
        mhd_file_path = os.path.join(
            "data/original_data/", mhd_file_name + f"_{name}.mhd"
        )
        image = sitk.ReadImage(mhd_file_path)
        image_array = sitk.GetArrayFromImage(image)
        image_lists.append(image_array)

    for i in range(0, image_array.shape[0], 64):
        for j in range(0, image_array.shape[1], 64):
            inf_patch = image_lists[0][i : i + 64, j : j + 64]
            sr_patch = image_lists[1][i : i + 64, j : j + 64]
            hr_patch = image_lists[2][i : i + 64, j : j + 64]
            patch_all_img = np.hstack((inf_patch, hr_patch, sr_patch))
            patch_all_img = patch_all_img.astype(np.uint8)
            patch_all_img = Image.fromarray(patch_all_img)
            patch_all_img.save(f"data/{mhd_file_name}/{i}_{j}.png")

        # output_file_path = os.path.join(
        #     "data/mhd", mhd_file_name + f"_all_{difine_point}.mhd"
        # )
        # sitk.WriteImage(patch_all_img_sitk, output_file_path)
        # save image as png


if __name__ == "__main__":
    create_patch_img("0_4")
    create_patch_img("0_5")
