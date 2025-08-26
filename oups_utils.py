import os
import shutil
from pathlib import Path


data_1 = "yolo_clean_less_empty"

data_2 = "yolo_preprocess2_less_empty"

src = "sentinel"


# from PIL import Image
# import os
# from glob import glob
# # Dossier contenant les TIFF
# image_dir = "datasets/sentinel/images"  # change selon besoin
# image_paths = glob(os.path.join(image_dir, "*.tif"))
# for path in image_paths:
#     img = Image.open(path)
#     if img.mode == "L":  # grayscale = 1 canal
#         # Dupliquer la bande 3 fois pour faire du "faux RGB"
#         img_rgb = Image.merge("RGB", (img, img, img))
#         img_rgb.save(path)  # écrase le TIFF d'origine avec version 3 canaux
#     else:
#         print(f"{path} déjà en {img.mode}, ignoré.")



for t  in ["images","labels"]:
    for split in ['val','train']:

        for file in os.listdir(Path("datasets",data_1,t,split)):
            path = Path("datasets",data_1,t,split,file)

            if t == "images":

                path_src = Path("datasets",src ,path.parts[-3],file )
            else :
                
                path_src = Path("datasets",data_1 ,t,split,file )


            path_dst = Path("datasets",data_2 ,t,split,file)

            shutil.copyfile(path_src,path_dst)
