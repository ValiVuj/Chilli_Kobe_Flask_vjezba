import glob
import cv2
import shutil 
from PIL import Image
import os
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array,load_img
from sklearn.datasets import load_sample_image


# old_extension="jpeg"
# new_extension="jpg"

# file_counter = 0
# image_paths = glob.glob(r"dataset/*/*.jpeg", recursive=True)

# for file in image_paths:
#         root, ext = file.split('.')
#         if ext == old_extension:
#                 new_path = root + "." + new_extension
#                 #print(file, new_path)
#                 os.rename(file, new_path)
#                 file_counter+=1
#                 print(file_counter)


image_paths_all_jpg = glob.glob(r"dataset/*/*.jpg", recursive=True)
i=0
for image_file in image_paths_all_jpg:
        img=Image.open(image_file)
        i+= 1
        print (i,") ",image_file,", resolution: ",img.size[0],"x",img.size[1])
...            


# import pandas as pd
# import matplotlib.pyplot  as plt
# from PIL import Image
# from pathlib import Path
# import imagesize
# import numpy as np

# # Get the Image Resolutions
# imgs = [img.name for img in Path(str(image_paths_all_jpg)).iterdir() if img.suffix == ".jpg"]
# img_meta = {}
# for f in imgs: img_meta[str(f)] = imagesize.get(str(image_paths_all_jpg+f))

# # Convert it to Dataframe and compute aspect ratio
# img_meta_df = pd.DataFrame.from_dict([img_meta]).T.reset_index().set_axis(['FileName', 'Size'], axis='columns', inplace=False)
# img_meta_df[["Width", "Height"]] = pd.DataFrame(img_meta_df["Size"].tolist(), index=img_meta_df.index)
# img_meta_df["Aspect Ratio"] = round(img_meta_df["Width"] / img_meta_df["Height"], 2)

# print(f'Total Nr of Images in the dataset: {len(img_meta_df)}')
# img_meta_df.head()