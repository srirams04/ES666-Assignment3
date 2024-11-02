import cv2
import numpy as np

def detect_and_match_keypoints(img1, img2):
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

def compute_homography_dlt(points1, points2):
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

def ransac_homography(points1, points2, num_iterations=1000, threshold=5.0):
    max_inliers = 0
    best_homography = None

    for _ in range(num_iterations):
        # Randomly select 4 points for minimal DLT
        idx = np.random.choice(len(points1), 4, replace=False)
        pts1_sample = points1[idx]
        pts2_sample = points2[idx]

        # Compute homography with the 4-point sample
        H = compute_homography_dlt(pts1_sample, pts2_sample)

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

def find_homography(img1, img2):
    # Detect and match keypoints
    points1, points2 = detect_and_match_keypoints(img1, img2)
    
    # Apply RANSAC to find the best homography matrix
    homography_matrix = ransac_homography(points1, points2)
    
    return homography_matrix

def to_mtx(img):
    H,V,C = img.shape
    mtr = np.zeros((V,H,C), dtype='int')
    for i in range(img.shape[0]):
        mtr[:,i] = img[i]
    
    return mtr

def to_img(mtr):
    V,H,C = mtr.shape
    img = np.zeros((H,V,C), dtype='int')
    for i in range(mtr.shape[0]):
        img[:,i] = mtr[i]
        
    return img

def warpPerspective(img, M, dsize):
    mtr = to_mtx(img)
    R,C = dsize
    dst = np.zeros((R,C,mtr.shape[2]))
    for i in range(mtr.shape[0]):
        for j in range(mtr.shape[1]):
            res = np.dot(M, [i,j,1])
            i2,j2,_ = (res / res[2] + 0.5).astype(int)
            if i2 >= 0 and i2 < R:
                if j2 >= 0 and j2 < C:
                    dst[i2,j2] = mtr[i,j]
    
    return to_img(dst)


def warp_image(img1, img2, H):
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
    output_img[translation_dist[1]:h2 + translation_dist[1], translation_dist[0]:w2 + translation_dist[0]] = img2
    
    return output_img

def interpolate_pixel(img, x, y):
    # Get the pixel intensity by bilinear interpolation
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, img.shape[1] - 1), min(y0 + 1, img.shape[0] - 1)
    
    # Bilinear interpolation weights
    a, b = x - x0, y - y0
    interp_value = (
        (1 - a) * (1 - b) * img[y0, x0] +
        a * (1 - b) * img[y0, x1] +
        (1 - a) * b * img[y1, x0] +
        a * b * img[y1, x1]
    )
    return interp_value

def stitch_images(img1, img2):
    # Detect and match keypoints
    points1, points2 = detect_and_match_keypoints(img1, img2)
    
    # Find the best homography matrix using RANSAC
    H = ransac_homography(points1, points2)
    
    # Warp img1 to match img2's perspective and create the stitched image
    stitched_img = warp_image(img1, img2, H)
    
    return stitched_img

# Load the images in color
img1 = cv2.imread('STC_0033.jpg')
img2 = cv2.imread('STD_0034.jpg')

    
# Stitch the images together
stitched_result = stitch_images(img1, img2)

# Show and save the result
cv2.imshow("Stitched Image", stitched_result)
cv2.imwrite("stitched_output.jpg", stitched_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
