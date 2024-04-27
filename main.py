import cv2
import numpy as np
import os
import sys


def dewarp(img):
    grey_blur_img = cv2.GaussianBlur(grey_img, (5, 5), 0)
    _, thresh = cv2.threshold(grey_blur_img, 0, 256, cv2.THRESH_TRIANGLE)
    outline_img = cv2.Canny(thresh, 50, 100, apertureSize=3)
    contours, _ = cv2.findContours(outline_img, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    img_contour = max(contours, key=cv2.contourArea)

    # https://stackoverflow.com/questions/58736927/how-to-find-
    # accurate-corner-positions-of-a-distorted-rectangle-from-blurry-image
    perimeter = 0.05 * cv2.arcLength(img_contour, True)
    approx = cv2.approxPolyDP(img_contour, perimeter, True)
    approx_float = approx.reshape(4, 2).astype(np.float32)
    destination_pts = np.array([[256, 0], [0, 0], [0, 256], [256, 256]],
                               dtype=np.float32)

    transform_mat = cv2.getPerspectiveTransform(approx_float, destination_pts)
    img = cv2.warpPerspective(img, transform_mat, (256, 256))

    return img


def denoise(img):
    # Colour channels have varying levels of noise, so treated separately.
    B, G, R = cv2.split(img)

    # For each colour channel, you isolate the noise by thresholding, then find
    # contours. The largest contours tend to be the body or ribcage, so you
    # remove them. Finally you end up with a mask of most of the noise, where
    # you can use inpainting to remove it. To remove the finer noise, apply
    # fastNlMeansDenoising. The technique above allows for less aggressive
    # fastNlMeansDenoising, so edges are preserved better.
###############################################################################
    _, B_thresh = cv2.threshold(B, 170, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    B_contours, _ = cv2.findContours(B_thresh, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
    B_largest_contours = sorted(B_contours, key=cv2.contourArea,
                                reverse=True)[:5]

    for contour in B_largest_contours:
        cv2.drawContours(B_thresh, [contour], -1, (0,), thickness=cv2.FILLED)

    B = cv2.inpaint(B, B_thresh, 1, cv2.INPAINT_TELEA)
    B = cv2.fastNlMeansDenoising(B, None, 30, 5, 40)
###############################################################################
    _, G_thresh = cv2.threshold(G, 160, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    G_contours, _ = cv2.findContours(G_thresh, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
    G_largest_contours = sorted(G_contours, key=cv2.contourArea,
                                reverse=True)[:15]

    for contour in G_largest_contours:
        cv2.drawContours(G_thresh, [contour], -1, (0,), thickness=cv2.FILLED)

    G = cv2.inpaint(G, G_thresh, 1, cv2.INPAINT_TELEA)
    G = cv2.fastNlMeansDenoising(G, None, 25, 4, 40)
###############################################################################
    _, R_thresh = cv2.threshold(R, 160, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    R_contours, _ = cv2.findContours(R_thresh, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
    R_largest_contours = sorted(R_contours, key=cv2.contourArea,
                                reverse=True)[:6]

    for contour in R_largest_contours:
        cv2.drawContours(R_thresh, [contour], -1, (0,), thickness=cv2.FILLED)

    R = cv2.inpaint(R, R_thresh, 1, cv2.INPAINT_TELEA)
    R = cv2.fastNlMeansDenoising(R, None, 25, 4, 30)
###############################################################################
    img = cv2.merge([B, G, R])

    # Could use median blur or bilateral filter to remove graininess
    # but after experimenting, edge preservation is more important
    return img


def colour_corr(img):
    B, G, R = cv2.split(img)

    # Tried a variety of methods, but manually adjusting
    # values worked the best. Analysis of histogram showed
    # that green was under-represented, and the image on the
    # pdf appeared to have a lot more green to them.
    G = cv2.add(G, 40)
    G = np.clip(G, 0, 255)

    img = cv2.merge([B, G, R])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(img)

    S = cv2.multiply(S, 3)
    H = cv2.multiply(H, 1)
    V = cv2.multiply(V, 1)

    img = cv2.merge([H, S, V])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    B, G, R = cv2.split(img)

    G = cv2.add(G, -40)
    G = np.clip(G, 0, 255)

    img = cv2.merge([B, G, R])

    return img


def contrast_brightness(img):

    # Applying CLAHE to the L channel of the LAB colour space
    # to increase contrast
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    img = cv2.merge([L, A, B])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    return img


def custom_inpaint(img):
    # My inpainting method relies that the human body is mostly symmetrical.
    # Knowing this, I can reflect the left side and apply a mask to fill in
    # the black circle. After filling in the circle, blending and colour
    # replacement is applied to make the image more natural. The blending
    # removes the sudden change in pixel values, when the image shifts from
    # the original one to the inpainted part. Sometimes, part of the background
    # is captured, so to fix this, I artificially change the values of the dark
    # blue in the inpainted area. A problem however is that sometimes the mask
    # should capture a bit of the background. To sort this, you can determine
    # whether the mask lies fully within the body or not. If it does, then
    # the dark blue should be replaced.

    # At first the mask is created by getting black pixels.
    # # https://www.tutorialspoint.com/how-to-mask-an-image-in-opencv-python#:
    # ~:text=We%20can%20apply%20a%20mask,of%20color%20values%20in%20HSV.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    l_black = np.array([0, 0, 0])
    u_black = np.array([180, 255, 12])
    mask = cv2.inRange(img, l_black, u_black)
    mask[:10, :] = 0
    mask[-10:, :] = 0
    mask[:, :10] = 0
    mask[:, -10:] = 0
    mask = cv2.resize(mask, (256, 256))

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # The mask is dilated to ensure that the area is fully covered
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, dilation_kernel, iterations=3)

    # Here we want to see if the mask is fully within the body,
    # or if it is on the edge. If the mask is on an edge of the body,
    # then we want our circularity test to fail, because we don't want
    # to replace the dark blue.
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(grey_img, (5, 5), 0)

    # To capture the true area of the mask, we create two binary thresholds.
    # One captures a lot of information, to ensure that the whole mask is
    # captured, whilst the other captures solely the mask (the borders of which
    # have been shrunk because we don't want to capture any extra detail).
    # NOTE - only using black pixels results in missing values, and if
    # dilated mask is used, then mask can spill out of the body.
    _, thresh_circle_full = cv2.threshold(blur_img, 0, 256, cv2.THRESH_OTSU
                                          + cv2.THRESH_BINARY_INV)
    _, thesh_circle_small = cv2.threshold(blur_img, 10, 150,
                                          cv2.THRESH_BINARY_INV)

    contours_small, _ = cv2.findContours(thesh_circle_small, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_NONE)
    contours_small = max(contours_small, key=cv2.contourArea)

    contours_full, _ = cv2.findContours(thresh_circle_full, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)

    true_mask = np.zeros_like(thresh_circle_full)

    first_contour_point = contours_small[0][0]
    point = (float(first_contour_point[0]), float(first_contour_point[1]))

    for contour in contours_full:
        # Checks if smaller circle is within the larger circle
        if cv2.pointPolygonTest(contour, point, False) >= 0:
            # Add the larger circle to true_mask
            cv2.drawContours(true_mask, [contour], -1, (255),
                             thickness=cv2.FILLED)

    # We have now captured our 'true mask'. This mask captures only the circle
    # if it is fully within the body, else it will capture the circle plus
    # the rest of the background.
    # We now want to understand if the mask is fully within or not. You can do
    # this by or-ing the true mask with the inverse of the more inaccurate mask
    # defined at the beginning. It does not matter if the original mask is not
    # completely accurate, as we just want to test for circularity.
    semi_circle = cv2.bitwise_or(true_mask, cv2.bitwise_not(mask))
    semi_circle = cv2.bitwise_not(semi_circle)
    # semi_circle is now either a ring or an arc.

    blurred_semi = cv2.GaussianBlur(semi_circle, (5, 5), 0)
    thresh_semi = cv2.adaptiveThreshold(blurred_semi, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

    canny_semi_circle = cv2.Canny(thresh_semi, 50, 100)
    semi_circle_contours, _ = cv2.findContours(canny_semi_circle,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
    semi_circle_c = max(semi_circle_contours,  key=cv2.contourArea)

    # Need radius to measure circularity
    (_, _), radius = cv2.minEnclosingCircle(semi_circle_c)
    radius = int(radius)

    # Circularity is measured to potentially fix dark blue areas that get
    # inpainted. If the circularity is confident enough, it means that the
    # mask is fully inside the body, hence the dark blue background should
    # not be there.
    circularity = ((4 * np.pi * cv2.contourArea(semi_circle_c))
                   / (cv2.arcLength(semi_circle_c, True) ** 2))

    circle_flag = False
    if circularity > 0.8:
        circle_flag = True

    # Getting the reflected image.
    midpoint = 256 // 2
    left_half = img[:, :midpoint]
    flipped_left = cv2.flip(left_half, 1)
    black_img = np.zeros_like(flipped_left)
    reflected_img = np.concatenate((black_img, flipped_left), axis=1)

    # Changing the values that lie in the mask to those
    # of the reflected image.
    cut_out = np.where(mask == 255)
    img[cut_out] = reflected_img[cut_out]

    if circle_flag:
        # Replacing the background as previously explained
        l_bound_replace_blue = np.array([170, 0, 0])
        u_bound_replace_blue = np.array([240, 205, 66])

        colour_mask = cv2.inRange(img, l_bound_replace_blue,
                                  u_bound_replace_blue)
        combined_mask = cv2.bitwise_and(colour_mask, mask)

        img[combined_mask == 255] = np.array([221, 213, 0])

    # To blend the surrounding area with the inpainted area,
    # a ring mask is created by dilating and eroding the original mask,
    # followed by xoring the two.
    dilation_kernel_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erode_kernel_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tmp_mask_dilated = cv2.dilate(mask, dilation_kernel_ring, iterations=3)
    tmp_mask_eroded = cv2.erode(mask, erode_kernel_ring, iterations=3)
    ring_mask = cv2.bitwise_xor(tmp_mask_dilated, tmp_mask_eroded)

    # The ring mask is then used to create a gradient mask.
    dist_transform = cv2.distanceTransform(ring_mask, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(dist_transform, None, 0, 1.0,
                                   cv2.NORM_MINMAX)
    adjusted_dist_transform = np.power(dist_transform, 1)
    # Hardness of gradient can be adjusted by changing the power value.
    gradient_ring_mask = adjusted_dist_transform * 255
    gradient_ring_mask = gradient_ring_mask.astype(np.uint8)

    # To blend the image well, a gaussian blur is applied,
    # such that the most inner parts of the ring are the most
    # blurred, in order to hide the sudden change in pixel values.
    blurred_img = cv2.GaussianBlur(img, (9, 9), 0)
    mask_blur = cv2.bitwise_and(blurred_img, blurred_img,
                                mask=gradient_ring_mask)

    img = np.where(gradient_ring_mask[:, :, None] != 0, mask_blur, img)

    return img


imgs = []
arg = sys.argv

if not os.path.exists(f'/Results'):
    os.makedirs(f'/Results')

for file in os.listdir(arg[1]):

    if file == '.DS_Store':
        continue

    img = cv2.imread(arg[1] + file)

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = dewarp(img)

    img = custom_inpaint(img)

    img = denoise(img)

    img = colour_corr(img)

    img = contrast_brightness(img)

    cv2.imwrite(f'/Results/{file}.jpg', img)
