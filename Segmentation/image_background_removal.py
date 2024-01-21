import cv2
import numpy as np


#== Parameters =======================================================================
BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) # In BGR format


#== Processing =======================================================================

#-- Read image -----------------------------------------------------------------------
#img = cv2.imread('/Users/d065887/Documents/SAP/Projects/fun/sample_images/shoe2.jpg')
#img = cv2.imread('/Users/d065887/Documents/SAP/Projects/fun/sample_images/658.jpg')
#img = cv2.imread("/Users/d065887/Documents/SAP/Projects/fun/shoes_2/ankle boots/best-boots Damen Plateau Stiefelette Chelsea Boots.jpg")
#img = cv2.imread("/Users/d065887/Documents/SAP/Projects/fun/sample_images/sneaker.jpg")
img = cv2.imread("/Users/d065887/Documents/SAP/Projects/fun/sample_images/boot2.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#-- Edge detection -------------------------------------------------------------------
edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)

#-- Find contours in edges, sort by area ---------------------------------------------
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
x,y,w,h = cv2.boundingRect(max_contour[0])
print((x,y,w,h))

#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# Mask is black, polygon is white
mask = np.zeros(edges.shape)
cv2.fillConvexPoly(mask, max_contour[0], (255))

#-- Smooth mask, then blur it --------------------------------------------------------
mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

#-- Blend masked img into MASK_COLOR background --------------------------------------
mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
img         = img.astype('float32') / 255.0                 #  for easy blending

masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

# split image into channels
c_red, c_green, c_blue = cv2.split(img)

# merge with mask got on one of a previous steps
img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))
img_a = img_a[y:y+h, x:x+w]

# save to disk
#cv2.imwrite('/Users/d065887/Documents/SAP/Projects/fun/shoe2_trans.jpg', img_a*255)
cv2.imwrite('/Users/d065887/Documents/SAP/Projects/fun/sample_images/boot2_trans.png', img_a*255)

#cv2.imshow('img', img_a)                                   # Display
#cv2.waitKey()




