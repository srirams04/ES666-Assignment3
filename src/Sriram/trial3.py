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
    num_points = points1.shape[0]
    A = []
    for i in range(num_points):
        x1, y1 = points1[i][0], points1[i][1]
        x2, y2 = points2[i][0], points2[i][1]
        A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])
    
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    
    return H / H[2, 2]

def ransac_homography(points1, points2, num_iterations=1000, threshold=5.0):
    max_inliers = 0
    best_homography = None

    for _ in range(num_iterations):
        idx = np.random.choice(len(points1), 4, replace=False)
        pts1_sample = points1[idx]
        pts2_sample = points2[idx]

        H = compute_homography_dlt(pts1_sample, pts2_sample)

        points1_homogeneous = np.hstack([points1, np.ones((points1.shape[0], 1))])
        projected_points = (H @ points1_homogeneous.T).T
        projected_points /= projected_points[:, 2].reshape(-1, 1)

        distances = np.linalg.norm(projected_points[:, :2] - points2, axis=1)
        inliers = distances < threshold
        num_inliers = np.sum(inliers)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_homography = H
    
    return best_homography

def find_homography(img1, img2):
    points1, points2 = detect_and_match_keypoints(img1, img2)
    homography_matrix = ransac_homography(points1, points2)
    return homography_matrix

def warp_image(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Define corner points of img1
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img1, H)
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    # Combine corners
    all_corners = np.concatenate((warped_corners, corners_img2), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min, -y_min]
    translation_matrix = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    
    # Warp img1 to the canvas size
    output_img = cv2.warpPerspective(img1, translation_matrix @ H, (x_max - x_min, y_max - y_min))
    
    # Create an alpha mask for img2 and overlay it
    overlay_img = np.zeros((output_img.shape[0], output_img.shape[1], 3), dtype=np.uint8)
    overlay_img[translation_dist[1]:h2 + translation_dist[1], translation_dist[0]:w2 + translation_dist[0]] = img2
    
    # Create a mask where img2 is located
    mask = np.zeros_like(output_img, dtype=np.uint8)
    mask[translation_dist[1]:h2 + translation_dist[1], translation_dist[0]:w2 + translation_dist[0]] = 255
    
    # Blend images
    combined = cv2.bitwise_or(output_img, overlay_img, mask=mask[:,:,0])
    
    return combined

def stitch_images(img1, img2):
    # Detect and match keypoints
    points1, points2 = detect_and_match_keypoints(img1, img2)
    
    # Find the best homography matrix using RANSAC
    H = ransac_homography(points1, points2)
    
    # Warp img1 to match img2's perspective and create the stitched image
    stitched_img = warp_image(img1, img2, H)
    
    return stitched_img

# Load the images in color
img1 = cv2.imread('STC_0033.jpg', cv2.IMREAD_UNCHANGED)  # Load with alpha if available
img2 = cv2.imread('STD_0034.jpg', cv2.IMREAD_UNCHANGED)

# Check if images have alpha channel and create masks accordingly
mask1 = np.zeros(img1.shape[:2], dtype=np.uint8)
mask2 = np.zeros(img2.shape[:2], dtype=np.uint8)

if img1.shape[2] == 4:  # RGBA
    mask1[img1[:,:,3] > 0] = 255
else:  # RGB
    mask1[img1 > 0] = 255  # Assuming non-black pixels are valid

if img2.shape[2] == 4:  # RGBA
    mask2[img2[:,:,3] > 0] = 255
else:  # RGB
    mask2[img2 > 0] = 255  # Assuming non-black pixels are valid

# Stitch the images together
stitched_result = stitch_images(img1, img2)

# Show and save the result
cv2.imshow("Stitched Image", stitched_result)
cv2.imwrite("stitched_output.jpg", stitched_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
