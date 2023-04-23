# Copyright 2023 (C) antillia.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#
# YOLOAnnotationGenerator.py
# 2023/04/23 : Toshiyuki Arai antillia.com

#    
#from email.mime import image
import sys
import os
import glob
import random
import shutil
import numpy as np

import traceback
import cv2
from PIL import Image


class YOLOAnnotationGenerator:
  def __init__(self, W=512, H=512):
    self.W = W
    self.H = H

  # dir = "./train/x
  # target = "./train" "./valid"
  def get_image_filepaths(self, images_dir ="./train/x"):
    pattern = images_dir + "/*.bmp"
    print("--- pattern {}".format(pattern))
    all_files  = glob.glob(pattern)
    image_filepaths = []
    for file in all_files:
      basename = os.path.basename(file)
      if basename.find("_") == -1:
        image_filepaths.append(file)
    return image_filepaths

  def get_mask_filepaths(self, image_filepath, mask_dir):
    basename = os.path.basename(image_filepath)
    name     = basename.split(".")[0]
    mask_filepattern  = mask_dir + "/" + name + "_*.bmp"
    mask_filepaths    = glob.glob(mask_filepattern)
    return mask_filepaths


  # target = "./train" or "./valid"
  def generate(self, input_dir, output_dir, debug=False):
    images_dir = input_dir + "/x/"
    masks_dir  = input_dir + "/y/"
    image_filepaths  = self.get_image_filepaths(images_dir)
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    for image_filepath in image_filepaths:
      basename = os.path.basename(image_filepath)
      name     = basename.split(".")[0]

      # 1 Create resize_image of size 512x512
      img_512x512 = self.create_resized_images(image_filepath)
      
      output_img_filepath = os.path.join(output_dir, name + ".jpg")
      # 2 Save the img_512x512 as a jpg file.
      img_512x512.save(output_img_filepath)
      print("=== Saved image_filepath {} as {}".format(image_filepath, output_img_filepath))

      # 3 Get some mask_filepaths corresponding to the image_filepath
      mask_filepaths = self.get_mask_filepaths(image_filepath, masks_dir)
      SP  = " "

      class_id    = 0
      annotations = []
      for mask_filepath in mask_filepaths:
        # 4 Create mask_image of size 512x512
        print("=== Create mask_image_512x512 from {}".format(mask_filepath))
        #PIL image format
        mask_img_512x512   = self.create_resized_images(mask_filepath, mask=True)

        # 5 Create a yolo annotation from the mask_img.
        (rcx, rcy, rw, rh) = self.create_yolo_annotation(mask_img_512x512)
        print(" rcx {} rcy {} rw {} rh {}".format(rcx, rcy, rw, rh))
        annotations.append( (rcx, rcy, rw, rh) )

      annotation_file = name + ".txt"
      annotation_file_path = os.path.join(output_dir, annotation_file)
      if debug:
        self.create_annotated_image(class_id, img_512x512,  annotations, basename, output_dir)
      
      NL = "\n"
      # 6 Create a yolo annotation file.
      with open(annotation_file_path, "w") as f:
          for annotation in annotations:
            (rcx, rcy, rw, rh) = annotation
            line = str(class_id ) + SP + str(rcx) + SP + str(rcy) + SP + str(rw) + SP + str(rh) 

            f.writelines(line + NL)
            print("---YOLO annotation {}".format(annotation))
      print("---Created annotation file {}".format(annotation_file_path))


  # Create a resized_512x512_image from each original file in image_filepaths
  def create_resized_images(self, image_filepath, mask=False):

    img = Image.open(image_filepath)
    print("---create_resized_512x512_images {}".format(image_filepath))
    
    #pixel = img.getpixel((128, 128))
    # We use the following fixed pixel for a background image.
    pixel = (207, 196, 208)
    if mask:
      pixel = (0, 0, 0)
    print("----pixel {}".format(pixel))
    w, h = img.size
    max = w
    if h > w:
      max = h
    if max < self.W:
      max = self.W
    # 1 Create a black background image
    background = Image.new("RGB", (max, max), pixel) # (0, 0, 0))
    #input("----HIT")
    # 2 Paste the original img to the background image at (x, y) position.
    print(img.format, img.size, img.mode)
    print(background.format, background.size, background.mode)

    x = int( (max - w)/2 )
    y = int( (max - h)/2 )
    background.paste(img, (x, y))

   
    background_512x512 = background.resize((self.W, self.H))
    if mask:
      background_512x512 = self.convert2WhiteMask(background_512x512)

    return background_512x512


  def create_yolo_annotation(self, pil_mask_img_512x512):
    mask_img = np.array(pil_mask_img_512x512)

    mask_img= cv2.cvtColor(mask_img,  cv2.COLOR_RGB2GRAY)
      
    H, W = mask_img.shape[:2]
       
    contours, hierarchy = cv2.findContours(mask_img, 
           cv2.RETR_EXTERNAL, 
           cv2.CHAIN_APPROX_SIMPLE)
       
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    x, y, w, h = cv2.boundingRect(contours)
    print("---x {} y {} w {} h {}".format(x, y, w, h))
    #Compute bouding box of YOLO format.
    cx = x + w/2
    cy = y + h/2
    #Convert to relative coordinates for YOLO annotations
    rcx = round(cx / W, 5)
    rcy = round(cy / H, 5)
    rw  = round( w / W, 5)
    rh  = round( h / H, 5)

    return (rcx, rcy, rw, rh)
        
    
  def convert2WhiteMask(self, image):
    w, h = image.size
    for y in range(h):
      for x in range(w):
        pixel = image.getpixel((x, y))
        if pixel != (0, 0, 0):
          pixel = (255, 255, 255) #White
          image.putpixel((x, y), pixel) 
    return image


  def create_annotated_image(self, class_id, _non_mask_img,  annotations, basename, output_subdir):
    _non_mask_img = np.array(_non_mask_img)

    GREEN  = (0, 255, 0)
    YELLOW = (0, 255, 255)
    output_dir_annotated = os.path.join(output_subdir, "annotated")
    if not os.path.exists(output_dir_annotated):
      os.makedirs(output_dir_annotated)
    for annotation in annotations:
      (rcx, rcy, rw, rh) = annotation
      cx = int (rcx * self.W)
      cy = int (rcy * self.H)
      w  = int (rw  * self.W)
      h  = int (rh  * self.H)
      x  = int (cx - w/2)
      y  = int (cy - h/2)
      _non_mask_img = cv2.rectangle(_non_mask_img , (x, y), (x+w, y+h), GREEN, 2)
      """
      cv2.putText(_non_mask_img,
              text      = str(class_id),
              org       = (x, y-30),
              fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
              fontScale = 0.5,
              color     = YELLOW,
              thickness = 2)
      """
      ouput_image_file_annotated = os.path.join(output_dir_annotated, basename)
      cv2.imwrite(ouput_image_file_annotated, _non_mask_img)
      print("--- create a annotated image file {}".format(ouput_image_file_annotated))



"""
INPUT:

./TCIA_SegPC_dataset
├─train
└─valid


Output:
./YOLO
├─train
└─valid


categories [0]
"MultipleMyeloma"        = 0
"""

if __name__ == "__main__":
  try:      
    # create Ovrian UltraSound Images OUS_augmented_master_512x512 dataset train, valid 
    # from the orignal Dataset_.

    input_dir   = "./TCIA_SegPC_dataset"
    # For simplicity, we have renamed the folder name from the original "validation" to "valid" 
    datasets    = ["train", "valid"]
    output_dir  = "./YOLO"

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    annotation= YOLOAnnotationGenerator(W=512, H=512)
    debug = True
    for dataset in datasets:
      input_subdir  = os.path.join(input_dir, dataset)
      output_subdir = os.path.join(output_dir, dataset)

      annotation.generate(input_subdir, output_subdir, debug=debug)
  except:
    traceback.print_exc()

      