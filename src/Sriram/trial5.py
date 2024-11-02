import cv2
import numpy as np
import glob

def cylindrical_warp(img, focal_length):
    """Warp an image into cylindrical coordinates."""
    h, w = img.shape[:2]
    cylinder_img = np.zeros_like(img)
    
    # Define the center of the image
    center_x, center_y = w // 2, h // 2

    # Loop through each pixel in the output cylindrical image
    for y in range(h):
        for x in range(w):
            # Convert to cylindrical coordinates
            theta = (x - center_x) / focal_length
            h_ = (y - center_y) / focal_length
            X = np.sin(theta)
            Y = h_
            Z = np.cos(theta)
            
            # Map cylindrical coordinates back to original coordinates
            x_original = int(focal_length * X / Z + center_x)
            y_original = int(focal_length * Y / Z + center_y)
            
            # Check if the mapped coordinates are within bounds
            if 0 <= x_original < w and 0 <= y_original < h:
                cylinder_img[y, x] = img[y_original, x_original]
    
    return cylinder_img

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

def warpCylindricalPerspective(img, K, output_size):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K, constrained to output_size."""
    h_, w_ = img.shape[:2]
    output_h, output_w = output_size
    
    # Pixel coordinates
    y_i, x_i = np.indices((h_, w_))
    X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)  # Convert to homogeneous coordinates
    
    # Inverse intrinsic matrix to normalize coordinates
    Kinv = np.linalg.inv(K)
    X = Kinv.dot(X.T).T  # Normalized coordinates
    
    # Calculate cylindrical coordinates (sin(theta), h, cos(theta))
    A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w_ * h_, 3)
    B = K.dot(A.T).T  # Project back to image-pixels plane
    
    # Back from homogeneous coordinates
    B = B[:, :-1] / B[:, [-1]]
    
    # Offset to center the warped image in the output canvas
    x_offset = (output_w - w_) // 2
    y_offset = (output_h - h_) // 2
    B[:, 0] += x_offset
    B[:, 1] += y_offset
    
    # Ensure warp coordinates stay within bounds of output_size
    B[(B[:, 0] < 0) | (B[:, 0] >= output_w) | (B[:, 1] < 0) | (B[:, 1] >= output_h)] = -1
    B = B.reshape(h_, w_, -1)
    
    # Convert image to RGBA for transparent borders
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # Warp the image according to cylindrical coordinates
    cylindrical_img = cv2.remap(img_rgba, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), 
                                cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)
    
    # Resize output to match the specified dimensions (output_size)
    return cylindrical_img[:output_h, :output_w]


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
    output_img = warpCylindricalPerspective(img1, translation_matrix @ H, (x_max - x_min, y_max - y_min))
    
    # Overlay img2 on the stitched image using translation
    output_img[translation_dist[1]:h2 + translation_dist[1], translation_dist[0]:w2 + translation_dist[0]] = img2
    
    return output_img

# def warp_image(img1, img2, H, focal_length):
#     """Stitch two cylindrical images using the homography matrix."""
#     # Warp both images into cylindrical coordinates
#     img1_cylindrical = img1
#     img2_cylindrical = img2

#     # Get dimensions of cylindrical images
#     h1, w1 = img1_cylindrical.shape[:2]
#     h2, w2 = img2_cylindrical.shape[:2]

#     # Warp the corners of img1 to find the size of the stitched canvas in cylindrical space
#     corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
#     warped_corners = cv2.perspectiveTransform(corners_img1, H)
#     corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
#     # Combine all corners to get the bounding box
#     all_corners = np.concatenate((warped_corners, corners_img2), axis=0)
#     [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
#     [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

#     # Calculate translation to fit everything in positive space
#     translation_dist = [-x_min, -y_min]
#     translation_matrix = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

#     # Initialize output canvas based on cylindrical projection
#     output_img = np.zeros((y_max - y_min, x_max - x_min, img1.shape[2]), dtype=img1.dtype)
    
#     # Place img1_cylindrical on the output canvas
#     output_img[translation_dist[1]:translation_dist[1] + h1, translation_dist[0]:translation_dist[0] + w1] = img1_cylindrical

#     # Overlay img2_cylindrical with the translation and homography applied
#     for y in range(h2):
#         for x in range(w2):
#             if np.any(img2_cylindrical[y, x] > 0):  # Ignore black pixels
#                 # Apply homography to cylindrical coordinates
#                 p = np.array([x, y, 1.0])
#                 p_transformed = translation_matrix @ H @ p
#                 x_new, y_new = (p_transformed[:2] / p_transformed[2]).astype(int)
#                 if 0 <= x_new < output_img.shape[1] and 0 <= y_new < output_img.shape[0]:
#                     output_img[y_new, x_new] = img2_cylindrical[y, x]

#     return output_img

def cylindricalWarp(img, K):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h_,w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    # warp the image according to cylindrical coords
    return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)


def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return image[y:y+h, x:x+w]
    return image


# Use the modified cylindrical_warp to prepare images for stitching
def stitch_images(img1, img2, focal_length):
    # Warp both images into cylindrical coordinates
    img1_cylindrical = cylindrical_warp(img1, focal_length)
    img2_cylindrical = cylindrical_warp(img2, focal_length)
    
    # Detect and match keypoints in cylindrical images
    points1, points2 = detect_and_match_keypoints(img1_cylindrical, img2_cylindrical)
    
    # Find homography matrix using RANSAC
    H = ransac_homography(points1, points2)
    
    # Stitch images using the homography on cylindrical images
    stitched_img = warp_image(img1_cylindrical, img2_cylindrical, H)
    
    return stitched_img

def stitch_multiple_images(images, focal_length):
    reference_idx = len(images) // 2
    # reference_idx = 0
    print(reference_idx, len(images))
    panorama = images[reference_idx]
    
    # for i in range(1, len(images)):
    #     panorama = stitch_images(images[reference_idx+i], panorama)
    #     panorama = crop_black_borders(panorama)
    #     cv2.imwrite(f"stitched_mix_{i}.jpg", panorama)

    for i in range(1, len(images)-reference_idx):
        panorama = stitch_images(images[reference_idx-i], panorama, focal_length)
        panorama = crop_black_borders(panorama)
        cv2.imwrite(f"stitched_mix_{i}_left.jpg", panorama)
        panorama = stitch_images(images[reference_idx+i], panorama, focal_length)
        panorama = crop_black_borders(panorama)
        cv2.imwrite(f"stitched_mix_{i}.jpg", panorama)

    if len(images) % 2 == 0:
        panorama = stitch_images(images[0], panorama, focal_length)
        panorama = crop_black_borders(panorama)

    return panorama

# Load images in color

# Define the path to your folder
folder_path = 'I3/*'

# Use glob to get all image file paths in the folder
image_files = glob.glob(folder_path)

# Initialize an empty list to store the images
images = []

# Loop through each file path, read the image and append it to the list
for file in image_files:
    img = cv2.imread(file)
    if img is not None:
        images.append(img)
        
print(images)

# Set focal length (approximate based on the camera and image dimensions)
focal_length = 700  # Adjust based on your camera setup

# Stitch the images into a panorama
stitched_panorama = stitch_multiple_images(images, focal_length)

# Show and save the result
cv2.imwrite("stitched_curv_I3.jpg", stitched_panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
