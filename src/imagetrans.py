import numpy as np
import cv2

#Config
s_thresh=(150, 255)
sx_thresh=(40, 120)
l_thresh = (40, 255)
labl = (0,95,155)
labh = (255,160,255)
    

def apply_sobel_x(channel):
    # Sobel x
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    return sxbinary

def to_binary_img(img):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sxbinary = apply_sobel_x(l_channel)
    
    # Threshold saturation channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
     # Threshold lightness channel
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
   
      
    white_binary = (sxbinary | s_binary) & l_binary
    
    #Convert to Lab for Yellow filtering
    lab = cv2.cvtColor(img,cv2.COLOR_RGB2Lab)
    low_threshold   = np.array(labl, dtype=np.uint8, ndmin=1)
    high_threshold  = np.array(labh, dtype=np.uint8, ndmin=1)
    yellow_mask = cv2.inRange(lab, low_threshold, high_threshold)
    yellow_binary = np.full_like(l_channel,1)
    yellow_binary = cv2.bitwise_and(yellow_binary,yellow_binary, mask=yellow_mask)
    
    # Stack each channel
    return (yellow_binary | white_binary)


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image
