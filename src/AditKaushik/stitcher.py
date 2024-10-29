import pdb
import glob
import cv2
import os
from src.JohnDoe import some_function
from src.JohnDoe.some_folder import folder_func
from typing import List, Tuple, Optional
import numpy as np

class CustomStitcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Status codes to match OpenCV's stitcher
        self.OK = 0
        self.ERR_NEED_MORE_IMGS = 1
        self.ERR_HOMOGRAPHY_EST_FAIL = 2
        self.ERR_CAMERA_PARAMS_ADJUST_FAIL = 3

    def detect_and_match_features(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
        """Detect SIFT features and match them between two images."""
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        print("%"*50,"Features Detected","%"*50)

        if des1 is None or des2 is None:
            return [], [], []

        des1 = np.float32(des1)
        des2 = np.float32(des2)

        try:
            matches = self.flann.knnMatch(des1, des2, k=2)
        except Exception as e:
            print(f"Error in feature matching: {e}")
            return kp1, kp2, []
        print("%"*50,"Features Matched","%"*50)

        good_matches = []
        try:
            for match_pair in matches:
                if len(match_pair) != 2:
                    continue
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        except Exception as e:
            print(f"Error in ratio test: {e}")
            return kp1, kp2, []
        print("%"*50,"Good Matches Found","%"*50)

        return kp1, kp2, good_matches

    def find_homography(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], 
                       matches: List[cv2.DMatch]) -> Optional[np.ndarray]:
        """Calculate homography matrix between two images using matched features."""
        if len(matches) < 4:
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def warp_and_blend(self, img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Warp and blend images using the homography matrix."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = cv2.perspectiveTransform(corners1, H)
        corners2 = corners2.reshape(-1, 2)

        min_x = min(np.min(corners2[:, 0]), 0)
        max_x = max(np.max(corners2[:, 0]), w2)
        min_y = min(np.min(corners2[:, 1]), 0)
        max_y = max(np.max(corners2[:, 1]), h2)

        offset = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        H_offset = offset @ H

        out_h = int(max_y - min_y)
        out_w = int(max_x - min_x)
        warped_img1 = cv2.warpPerspective(img1, H_offset, (out_w, out_h))

        img2_warped = np.zeros_like(warped_img1)
        
        x_offset = int(-min_x)
        y_offset = int(-min_y)
        
        y_start = max(0, y_offset)
        y_end = min(out_h, y_offset + h2)
        x_start = max(0, x_offset)
        x_end = min(out_w, x_offset + w2)
        
        src_y_start = max(0, -y_offset)
        src_x_start = max(0, -x_offset)
        src_y_end = src_y_start + (y_end - y_start)
        src_x_end = src_x_start + (x_end - x_start)
        
        img2_warped[y_start:y_end, x_start:x_end] = img2[src_y_start:src_y_end, src_x_start:src_x_end]

        # Simple alpha blending
        alpha = 0.5
        mask = np.zeros((out_h, out_w), dtype=np.float32)
        mask[y_start:y_end, x_start:x_end] = alpha

        result = warped_img1.copy()
        mask = mask[:, :, np.newaxis]
        result = (1 - mask) * warped_img1 + mask * img2_warped

        return result.astype(np.uint8)

    def stitch(self, images: List[np.ndarray]) -> Tuple[int, Optional[np.ndarray]]:
        """
        Stitch multiple images together.
        Returns:
            Tuple[int, Optional[np.ndarray]]: Status code and stitched image (if successful)
        """
        if len(images) < 2:
            return self.ERR_NEED_MORE_IMGS, None

        # Check if images are valid
        if not all(img is not None for img in images):
            return self.ERR_NEED_MORE_IMGS, None

        # Use first image as reference
        result = images[0]

        # Stitch each subsequent image
        print("num of images: ",len(images))
        for img in images[1:]:
            # Convert to grayscale for feature detection
            gray1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print("%"*50,"Images Converted to Gray","%"*50)

            # Detect and match features
            kp1, kp2, matches = self.detect_and_match_features(gray1, gray2)
            if not matches:
                return self.ERR_HOMOGRAPHY_EST_FAIL, None
            print("%"*50,"Matches Found: ",len(matches),"%"*50)

            # Find homography
            H = self.find_homography(kp1, kp2, matches)
            if H is None:
                return self.ERR_HOMOGRAPHY_EST_FAIL, None
            print("%"*50,"Homography Found","%"*50)

            # Warp and blend images
            result = self.warp_and_blend(result, img, H)
            print("%"*50,"Image Stitched","%"*50)

        return self.OK, result








class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        ####  Your Implementation here
        #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
        #### Just make sure to return final stitched image and all Homography matrices from here
        self.say_hi()
        self.do_something()
        self.do_something_more()

        some_function.some_func()
        folder_func.foo()

        # Collect all homographies calculated for pair of images and return
        homography_matrix_list =[]
        # Return Final panaroma
        #stitcher = cv2.Stitcher_create()
        stitcher = CustomStitcher()
        print("%"*50,"Stitcher object created","%"*50)
        status, stitched_image = stitcher.stitch([cv2.imread(im) for im in all_images])
        # stitched_image = cv2.imread(all_images[0])
        #####
        
        return stitched_image, homography_matrix_list 

    def say_hi(self):
        print('Hii From Jane Doe..')
    
    def do_something(self):
        return None
    
    def do_something_more(self):
        return None
