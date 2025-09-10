import cv2
import numpy as np
import matplotlib.pyplot as plt
def stitch_images_sift(image1_path, image2_path):
    # Read the images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Store good matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Extract location of good matches
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    
    # Use homography to warp image
    height, width = img1.shape[:2]
    result = cv2.warpPerspective(img2, h, (width * 2, height))
    result[0:height, 0:width] = img1
    
    return result


if __name__ == "__main__":
    # Replace with your image paths
    image1_path = "left.jpg"  # First camera image
    image2_path = "right.jpg" # Second camera image
    
    try:
        stitched_image = stitch_images_sift(image1_path, image2_path)
        
        # Display the result
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
        plt.title("Stitched Image")
        plt.axis('off')
        plt.show()
        
        # Save the result
        cv2.imwrite("stitched_result.jpg", stitched_image)
        print("Stitched image saved as 'stitched_result.jpg'")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure the image paths are correct and images are loaded properly.")