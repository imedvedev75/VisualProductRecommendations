import argparse
import numpy as np
import cv2
import os
import random

MAX_RATIO = 1.05


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-180, 180), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def generator(img, mask, bg, num, start, do_transform, output_path, image_name, out_stats):
    # rescale image if required
    if img.shape[0] >= bg.shape[0] or img.shape[1] >= bg.shape[1]:
        ratioy = img.shape[0] * 2 / bg.shape[0]
        ratiox = img.shape[1] * 2 / bg.shape[1]
        r = int(float(max(ratiox, ratioy)))
        img = cv2.resize(img, (img.shape[1] / r, img.shape[0] / r), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (img.shape[1] / r, img.shape[0] / r), interpolation=cv2.INTER_AREA)
    elif bg.shape[0] >= img.shape[0] * MAX_RATIO or bg.shape[1] >= img.shape[1] * MAX_RATIO:
        ratioy = int(bg.shape[0] / (img.shape[0] * MAX_RATIO))
        ratiox = int(bg.shape[1] / (img.shape[1] * MAX_RATIO))
        r = int(float(min(ratiox, ratioy)))
        if (r == 0):
            r = 1
        #print("%s\t%s\t%s\t%s" % (bg.shape[0], bg.shape[1], img.shape[0], img.shape[1]))
        bg = cv2.resize(bg, (bg.shape[1] / r, bg.shape[0] / r), interpolation=cv2.INTER_AREA)

    #print("image shape = %s" % ', '.join(map(str, img.shape)))
    #print("mask shape = %s" % ', '.join(map(str, mask.shape)))
    #print("background shape = %s" % ', '.join(map(str, bg.shape)))

    # generate outputs
    for k in range(num):
        k += start
        img_tmp = img
        mask_tmp = mask
        if do_transform:
            #img_tmp = randomHueSaturationValue(img_tmp,
            #                                  hue_shift_limit=(-50, 50),
            #                                   sat_shift_limit=(-5, 5),
            #                                   val_shift_limit=(-15, 15))
            img_tmp, mask_tmp = randomShiftScaleRotate(img_tmp, mask_tmp,
                                                       shift_limit=(0, 0),
                                                       #shift_limit=(-0.0625, 0.0625),
                                                       scale_limit=(-0.1, 0.1),
                                                       rotate_limit=(-90, 90))
            img_tmp, mask_tmp = randomHorizontalFlip(img_tmp, mask_tmp)

        # choose location
        y = random.randint(0, bg.shape[0] - img_tmp.shape[0])
        x = random.randint(0, bg.shape[1] - img_tmp.shape[1])

        # initialize output
        out_arr = np.array(bg, np.uint8)
        out_mask = np.zeros((out_arr.shape[0], out_arr.shape[1], 3), np.uint8)

        # populate output image and output mask
        for i in range(img_tmp.shape[0]):
            for j in range(img_tmp.shape[1]):
                if np.all(mask_tmp[i, j] == 255):
                    out_arr[y + i, x + j] = img_tmp[i, j]
                    out_mask[y + i, x + j] = [255, 255, 255]

        # write outputs
        fimage = os.path.join(output_path, "%s_%s.jpg" % (image_name, k))
        fmask = os.path.join(output_path, "%s_%s_mask.jpg" % (image_name, k))
        cv2.imwrite(fimage, out_arr)
        cv2.imwrite(fmask, out_mask)

        if out_stats is not None:
            out_stats.write("%s+++%s+++%s+++%s+++%s+++%s\n" % (fimage, x, y, x+img_tmp.shape[1]-1, y+img_tmp.shape[0]-1, "shoe"))


def generator_single():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, default="input_image",
                        help="input image file")
    parser.add_argument("--input_mask", type=str, default="input_mask",
                        help="input mask")
    parser.add_argument("--input_background", type=str, default="input_background",
                        help="input background")
    parser.add_argument("--output_path", type=str, default="output",
                        help="output folder")
    parser.add_argument("--num", type=int, default=1,
                        help="number of output images")
    parser.add_argument("--do_transform", type=bool, default=False,
                        help="flag indicating if transforming input image and input mask")
    args = parser.parse_args()

    input_image = args.input_image
    input_mask = args.input_mask
    input_background = args.input_background
    output_path = args.output_path
    num = args.num
    do_transform = args.do_transform

    # get image name
    image_name = input_image.split("/")[-1]
    image_name = image_name.split(".")[0]

    # read image
    img = cv2.imread(input_image)

    # read mask
    mask = cv2.imread(input_mask)

    # read background
    bg = cv2.imread(input_background)

    # generate image
    generator(img=img, mask=mask, bg=bg, num=num, start=0, do_transform=do_transform, output_path=output_path,
              image_name=image_name, out_stats=None)


def generator_batch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_path", type=str, default="input_image_path",
                        help="input image path")
    parser.add_argument("--input_background_path", type=str, default="input_background_path",
                        help="input background path")
    parser.add_argument("--output_path", type=str, default="output",
                        help="output folder")
    parser.add_argument("--num", type=int, default=1,
                        help="number of output images")
    parser.add_argument("--do_transform", type=bool, default=False,
                        help="flag indicating if transforming input image and input mask")
    parser.add_argument("--output_statistics", type=str, default="output_statistics",
                        help="output statistics")
    args = parser.parse_args()

    input_image_path = args.input_image_path
    input_background_path = args.input_background_path
    output_path = args.output_path
    num = args.num
    do_transform = args.do_transform
    output_statistics = args.output_statistics

    # read backgrounds
    bgs = []
    for input_background in os.listdir(input_background_path):
        if not input_background.endswith(".jpg"):
            continue
        bgs.append(cv2.imread(os.path.join(input_background_path, input_background)))

    # read images and masks and generate output
    with open(output_statistics, "w") as out_stats:
        for input_image in os.listdir(input_image_path):
            if input_image.endswith("_mask.jpg") or not input_image.endswith(".jpg"):
                continue

            # get image name
            input_image = os.path.join(input_image_path, input_image)
            image_name = input_image.split("/")[-1]
            image_name = image_name.split(".")[0]

            # read image
            img = cv2.imread(input_image)

            # read mask
            mask = cv2.imread(input_image.replace(".jpg", "_mask.jpg"))

            # generate image
            start = 0
            for bg in bgs:
                # generate images
                generator(img=img, mask=mask, bg=bg, num=num, start=start, do_transform=do_transform,
                          output_path=output_path, image_name=image_name, out_stats=out_stats)

                # update start
                start += num


if __name__ == "__main__":
    #generator_single()
    generator_batch()

