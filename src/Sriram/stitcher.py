import pdb
import glob
import numpy as np
import cv2
import os

class PanaromaStitcher():
    def __init__(self):
        pass
    
    def detect_and_match_keypoints(self, img1, img2):
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect keypoints and descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
        
        
        # Match descriptors using FLANN-based matcher
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply ratio test to filter out weak matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # Extract matched keypoints
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
        
        return points1, points2

    def compute_homography_dlt(self, points1, points2):
        # Number of points
        num_points = points1.shape[0]
        
        # Construct matrix A for DLT
        A = []
        for i in range(num_points):
            x1, y1 = points1[i][0], points1[i][1]
            x2, y2 = points2[i][0], points2[i][1]
            A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
            A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])
        
        A = np.array(A)
        
        # Perform SVD on A
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        
        # Normalize and return homography
        return H / H[2, 2]

    def ransac_homography(self, points1, points2, num_iterations=1000, threshold=5.0):
        max_inliers = 0
        best_homography = None

        for _ in range(num_iterations):
            # Randomly select 4 points for minimal DLT
            idx = np.random.choice(len(points1), 4, replace=False)
            pts1_sample = points1[idx]
            pts2_sample = points2[idx]

            # Compute homography with the 4-point sample
            H = self.compute_homography_dlt(pts1_sample, pts2_sample)

            # Apply homography to all points
            points1_homogeneous = np.hstack([points1, np.ones((points1.shape[0], 1))])
            projected_points = (H @ points1_homogeneous.T).T
            projected_points /= projected_points[:, 2].reshape(-1, 1)

            # Calculate distances
            distances = np.linalg.norm(projected_points[:, :2] - points2, axis=1)
            inliers = distances < threshold
            num_inliers = np.sum(inliers)

            # Update the best homography if more inliers are found
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_homography = H
        
        return best_homography


    def warp_image(self, img1, img2, H):
        # Get the dimensions of the first image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Define corner points of the first image
        corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        
        # Warp the corners of img1 to find the size of the stitched canvas
        warped_corners = cv2.perspectiveTransform(corners_img1, H)
        corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        
        # Combine all corners to get the bounding box
        all_corners = np.concatenate((warped_corners, corners_img2), axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # Calculate the translation matrix to shift the result into positive coordinates
        translation_dist = [-x_min, -y_min]
        translation_matrix = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
        
        # Warp img1 to the canvas size
        output_img = cv2.warpPerspective(img1, translation_matrix @ H, (x_max - x_min, y_max - y_min))
        
        # Overlay img2 on the stitched image using translation
        output_img[translation_dist[1]:h2 + translation_dist[1], translation_dist[0]:w2 + translation_dist[0]][cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) > 0] = img2[cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) > 0]
        
        return output_img


    def stitch_images(self, img1, img2):
        # Detect and match keypoints
        points1, points2 = self.detect_and_match_keypoints(img1, img2)
        
        # Find the best homography matrix using RANSAC
        H = self.ransac_homography(points1, points2)
        self.homography_matrix_list.append(H)

        # Warp img1 to match img2's perspective and create the stitched image
        stitched_img = self.warp_image(img1, img2, H)
        
        return stitched_img

    def crop_black_borders(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            return image[y:y+h, x:x+w]
        return image

    def stitch_multiple_images(self, images):
        reference_idx = len(images) // 2
        # reference_idx = 0
    
        panorama = images[reference_idx]
        
        # for i in range(1, len(images)):
        #     panorama = stitch_images(images[reference_idx+i], panorama)
        #     panorama = crop_black_borders(panorama)
        #     cv2.imwrite(f"stitched_mix_{i}.jpg", panorama)

        for i in range(1, len(images)-reference_idx):
            panorama = self.stitch_images(images[reference_idx-i], panorama)
            # panorama = crop_black_borders(panorama)
            cv2.imwrite(f"stitched_mix_{i}_left.jpg", panorama)
            panorama = self.stitch_images(images[reference_idx+i], panorama)
            panorama = self.crop_black_borders(panorama)
            cv2.imwrite(f"stitched_mix_{i}.jpg", panorama)

        if len(images) % 2 == 0:
            panorama = self.stitch_images(images[0], panorama)
            panorama = self.crop_black_borders(panorama)

        return panorama

    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))


        # Collect all homographies calculated for pair of images and return
        self.homography_matrix_list =[]
        
        # Initialize an empty list to store the images
        images = []

        # Loop through each file path, read the image and append it to the list
        for file in all_images:
            img = cv2.imread(file)
            if img is not None:
                images.append(img)
                
    

        # Stitch the images into a panorama
        stitched_image = self.stitch_multiple_images(images)

        
        return stitched_image, self.homography_matrix_list 
