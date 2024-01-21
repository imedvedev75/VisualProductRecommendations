import cv2
import numpy as np
import argparse
import os


# parameters
BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) # In BGR format


def fill_holes(im_th):
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out


def generate_mask(input_file, output_path, flag_output_org_img):
    # get image name
    image_name = input_file.split("/")[-1]
    image_name = image_name.split(".")[0]

    # read image
    img = cv2.imread(input_file)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # output original image if required
    if flag_output_org_img:
        cv2.imwrite(os.path.join(output_path, image_name + ".jpg"), img)

    # edge detection
    edges = cv2.Canny(img, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # find contours in edges, sort by area
    contour_info = []
    _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]
    x, y, w, h = cv2.boundingRect(max_contour[0])
    print((x, y, w, h))

    # clone image
    #img_cloned = img.copy()

    # draw contour
    #cv2.drawContours(img_cloned, contours, 0, (0, 255, 0), 3)
    #cv2.drawContours(img_cloned, [max_contour[0]], 0, (0, 255, 0), 3)
    #cv2.imwrite(os.path.join(output_path, image_name + "_contour.jpg"), img_cloned)

    # draw mask
    mask = np.zeros(img.shape, np.uint8)
    #cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    cv2.drawContours(mask, [max_contour[0]], -1, (255, 255, 255), -1)
    #mask[:, :, 0] = fill_holes(mask[:, :, 0])
    #mask[:, :, 1] = fill_holes(mask[:, :, 1])
    #mask[:, :, 2] = fill_holes(mask[:, :, 2])
    cv2.imwrite(os.path.join(output_path, image_name + "_mask.jpg"), mask)

    # flag do_further
    do_further = True
    if not do_further:
        return

    # create empty mask, draw filled polygon on it corresponding to largest contour
    # mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))
    cv2.imwrite(os.path.join(output_path, image_name + "_mask_polygon.jpg"), mask)

    # smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    cv2.imwrite(os.path.join(output_path, image_name + "_mask_smoothened.jpg"), mask)
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    # blend masked img into MASK_COLOR background
    mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
    img = img.astype('float32') / 255.0  # for easy blending
    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
    masked = (masked * 255).astype('uint8')  # Convert back to 8-bit
    cv2.imwrite(os.path.join(output_path, image_name + "_mask_blended.jpg"), masked)

    # split image into channels
    c_red, c_green, c_blue = cv2.split(img)

    # merge with mask got on one of a previous steps
    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))
    img_a = img_a[y:y + h, x:x + w, :]

    # save to disk
    cv2.imwrite(os.path.join(output_path, image_name + "_trans.jpg"), img_a * 255)


def generate_mask_single():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="input",
                        help="input image file")
    parser.add_argument("--output_path", type=str, default="output",
                        help="output folder")
    args = parser.parse_args()

    input_file = args.input_file
    output_path = args.output_path
    generate_mask(input_file=input_file, output_path=output_path, flag_output_org_img=True)


def generate_mask_batch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_path", type=str, default="input_image_path",
                        help="input image path")
    parser.add_argument("--output_path", type=str, default="output",
                        help="output folder")
    args = parser.parse_args()

    input_image_path = args.input_image_path
    output_path = args.output_path

    c = 0
    for input_image in os.listdir(input_image_path):
        if not input_image.endswith(".jpg"):
            continue

        c += 1
        input_image = os.path.join(input_image_path, input_image)
        generate_mask(input_file=input_image, output_path=output_path, flag_output_org_img=True)

    print(c)

if __name__ == "__main__":
    generate_mask_single()
    #generate_mask_batch()






