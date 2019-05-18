import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import PIL
import os
import xml.etree.ElementTree as ET
from shutil import copyfile
import shutil
import matplotlib.pyplot as plt

n_aug_per_img = 30
turn_to_1_label = 0


# ------------   define transformations   ----------------
def transform(image, bbs_oi):
    seq = iaa.Sequential([
        iaa.Scale((0.5, 0.5)),
        iaa.Multiply((0.7, 1.3)),  # change brightness, doesn't affect BBs
        iaa.Fliplr(0.3),
        iaa.Sometimes(
            0.7,
            iaa.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
            rotate=(-50, 50),  # rotate by -90 to +90 degrees
            shear=(-8, 8)# shear by -16 to +16 degrees
            )),
        iaa.GaussianBlur(sigma=(0, 1)),  # blur images with a sigma of 0 to 3.0
        iaa.Add((-20, 20), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
        iaa.Sometimes(
            0.5,
            iaa.CropAndPad(percent=(-0.15, 0.15))),
        iaa.Sometimes(
            0.2,
            iaa.CoarseDropout((0.0, 0.02), size_percent=(0.008, 0.03), per_channel=0.3))
    ])
    seq = seq.to_deterministic()
    image_aug = seq.augment_images([image])[0]
    bbs_aug = seq.augment_bounding_boxes([bbs_oi])[0]
    return image_aug, bbs_aug


xml_dir = 'C:/Users/Dean/Documents/GitHub/models/research/object_detection/AUGMENTATION/xml/'
img_dir = 'C:/Users/Dean/Documents/GitHub/models/research/object_detection/AUGMENTATION/img/'
files = os.listdir(xml_dir)

aug_dir = 'C:/Users/Dean/Documents/GitHub/models/research/object_detection/AUGMENTATION/img_aug/'
if not os.path.exists(aug_dir):
    os.makedirs(aug_dir)
else:
    shutil.rmtree(aug_dir, ignore_errors=True)
    os.makedirs(aug_dir)

xml_aug_dir = 'C:/Users/Dean/Documents/GitHub/models/research/object_detection/AUGMENTATION/xml_aug/'
if not os.path.exists(xml_aug_dir):
    os.makedirs(xml_aug_dir)
else:
    shutil.rmtree(xml_aug_dir, ignore_errors=True)
    os.makedirs(xml_aug_dir)

for k, file in enumerate(files):
    print("Processing file " + str(k+1))
    # load image corresponding to xml file as numpy array
    im = PIL.Image.open(img_dir + file.split(".")[0] + ".jpg")
    image = np.array(im)
    # Loop through all bounding boxes in current image
    tree = ET.parse(os.path.join(xml_dir + file))
    root = tree.getroot()
    bbs = list()
    for obj in root.iter('object'):  # bounding boxes
        for bb in obj.iter('bndbox'):
            x1 = int(bb.getchildren()[0].text)
            y1 = int(bb.getchildren()[1].text)
            x2 = int(bb.getchildren()[2].text)
            y2 = int(bb.getchildren()[3].text)
            bbs.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
    bbs_oi = ia.BoundingBoxesOnImage(bbs, shape=image.shape)

    for i in range(n_aug_per_img):
        image_aug, bbs_aug = transform(image, bbs_oi)

        # image with BBs before/after augmentation (shown below)
        # plt.ion()
        # plt.figure()
        # image_before = bbs_oi.draw_on_image(image, thickness=3)
        # plt.imshow(image_before)
        # plt.figure()
        # image_after = bbs_aug.draw_on_image(image_aug, thickness=3, color=[255, 255, 0])
        # plt.imshow(image_after)
        image_aug = PIL.Image.fromarray(image_aug.astype('uint8'), 'RGB')
        image_aug.save(aug_dir + file.split(".")[0] + "_aug" + str(i) + ".jpg")
        # Copy original xml file and edit bounding boxes values
        copyfile(xml_dir + file, xml_aug_dir + file.split(".")[0] + "_aug" + str(i) + '.xml')
        tree = ET.parse(os.path.join(xml_dir + file))
        root = tree.getroot()
        root.getchildren()[0].text = 'train'
        root.getchildren()[1].text = root.getchildren()[1].text.split(".")[0] + "_aug" + str(i) + ".jpg"
        width, height = image_aug.size
        root[4].getchildren()[0].text = str(width)
        root[4].getchildren()[1].text = str(height)
        for j, obj in enumerate(root.iter('object')):  # bounding boxes
            for bb in obj.iter('bndbox'):
                bb.getchildren()[0].text = str(bbs_aug.bounding_boxes[j].x1_int)
                bb.getchildren()[1].text = str(bbs_aug.bounding_boxes[j].y1_int)
                bb.getchildren()[2].text = str(bbs_aug.bounding_boxes[j].x2_int)
                bb.getchildren()[3].text = str(bbs_aug.bounding_boxes[j].y2_int)
        tree.write(xml_aug_dir + file.split(".")[0] + "_aug" + str(i) + '.xml')

print("Done!")
# plt.waitforbuttonpress()

# # image with BBs before/after augmentation (shown below)
# image_before = bbs_oi.draw_on_image(image, thickness=5)
# plt.imshow(image_before)
# plt.figure()
# image_after = bbs_aug.draw_on_image(image_aug, thickness=5, color=[0, 0, 255])
# plt.imshow(image_after)
# plt.waitforbuttonpress()

