class LaneFitter():

    def __init__(self, camera):
        if camera is None:
            raise Exception("Lane fitter needs a camera to undistort and warp")
        self.camera = camera
    def getfit(self, img, prev_left_fit=None, prev_right_fit=None):
        undistorted = self.camera.undistort(img)
        binary_img = to_binary_img(undistorted)
        warped_binary = self.camera.warp(binary_img)

        if prev_left_fit is not None:
            lf, rf, lx, ly, rx, ry = search_around_poly(warped_binary, prev_left_fit, prev_right_fit)
        if prev_left_fit is None:
            lf, rf, lx, ly, rx, ry, left_rectangles, right_rectangles =  fit_polynomial(warped_binary)

        fit_info = (lf, rf, lx, ly, rx, ry)
        debug_info = (binary_img, warped_binary)

        return fit_info, debug_info