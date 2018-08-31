import numpy as np
import glob
import cv2

# Define a class to receive the characteristics of each line detection
class Camera():
    def __init__(self):
        self.ret = None
        self.mtx = None
        self.M = None
        self.M_inv = None
        
        
    def calibrate(self, imgs_path, draw_chessboards=False):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        detected = []
        undetected = []
        shape = None

        images = glob.glob(imgs_path)
        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
            # If found, add object points, image points
            if ret == True:
                if shape == None:
                    shape = img.shape[1::-1]
                detected.append(fname)
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                if draw_chessboards:
                    img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                    cv2.imshow('img',img)
                    cv2.waitKey(500)
            else:
                undetected.append(fname)

        if draw_chessboards:
            cv2.destroyAllWindows()
            cv2.waitKey(500)
            
        if len(imgpoints) == 0:
            raise Exception('No chessboards detected on images. Glob: ' + imgs_path)
            
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
        
        self.dist = dist
        self.ret = ret
        self.mtx = mtx
        
        return (detected, undetected)
        
    def undistort(self, img):
        if self.mtx is None:
            raise Exception('Camera has not been calibrated.') 
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist
   
    def prepare_perspective(self, original_pts, dst_pts):
        self.M = cv2.getPerspectiveTransform(original_pts, dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(dst_pts, original_pts)

    def warp(self, img):
        warped = cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        return warped

    def unwarp(self, img):
        warped = cv2.warpPerspective(img, self.M_inv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        return warped
