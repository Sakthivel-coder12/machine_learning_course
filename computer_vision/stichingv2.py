import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_keypoints_and_matches(img1, img2, kp1, kp2, matches):
    """Visualize keypoints and matches between images"""
    # Draw keypoints on both images
    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Display results
    plt.figure(figsize=(20, 15))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
    plt.title('Image 1 Keypoints')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
    plt.title('Image 2 Keypoints')
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matches')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def simple_stitch_images(image1_path, image2_path):
    # Read the images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not load one or both images. Check the file paths.")
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector (faster and often works better for simple cases)
    orb = cv2.ORB_create(nfeatures=2000)
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Take only the top 50 matches
    good_matches = matches[:50]
    
    # Show keypoints and matches for debugging
    show_keypoints_and_matches(img1, img2, kp1, kp2, good_matches)
    
    # Check if we have enough good matches
    if len(good_matches) < 10:
        raise ValueError(f"Not enough good matches found: {len(good_matches)}")
    
    # Extract location of good matches
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
    
    for i, match in enumerate(good_matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    
    # Find homography
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    
    if H is None:
        raise ValueError("Homography matrix could not be computed.")
    
    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Get the canvas size
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    # Transform corners of image2
    warped_corners2 = cv2.perspectiveTransform(corners2, H)
    
    # Combine corners
    all_corners = np.concatenate((corners1, warped_corners2), axis=0)
    
    # Find the bounding box
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Translation matrix to keep both images in view
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], 
                             [0, 1, translation_dist[1]], 
                             [0, 0, 1]])
    
    # Warp the second image
    result_width = x_max - x_min
    result_height = y_max - y_min
    warped_img2 = cv2.warpPerspective(img2, H_translation.dot(H), (result_width, result_height))
    
    # Create result canvas
    result = warped_img2.copy()
    
    # Place the first image
    x_start = translation_dist[0]
    y_start = translation_dist[1]
    x_end = x_start + w1
    y_end = y_start + h1
    
    # Ensure we don't exceed bounds
    x_end = min(x_end, result_width)
    y_end = min(y_end, result_height)
    
    # Place img1 in the result
    result[y_start:y_end, x_start:x_end] = img1[0:y_end-y_start, 0:x_end-x_start]
    
    return result, warped_img2, len(good_matches)

def manual_stitch_with_crop(image1_path, image2_path, overlap_percent=0.3):
    """Manual stitching by cropping and simple concatenation"""
    # Read images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not load images")
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Calculate overlap width
    overlap_width = int(w1 * overlap_percent)
    
    # Crop images
    img1_cropped = img1[:, :-overlap_width//2]  # Remove some of the right side
    img2_cropped = img2[:, overlap_width//2:]   # Remove some of the left side
    
    # Simple horizontal concatenation
    result = np.hstack((img1_cropped, img2_cropped))
    
    return result

# Example usage
if __name__ == "__main__":
    # Replace with your image paths
    image1_path = "left.jpg"
    image2_path = "right.jpg"
    
    try:
        print("Attempting automatic stitching with feature matching...")
        result, warped_img2, num_matches = simple_stitch_images(image1_path, image2_path)
        
        print(f"Number of good matches found: {num_matches}")
        
        # Display results
        plt.figure(figsize=(20, 10))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(warped_img2, cv2.COLOR_BGR2RGB))
        plt.title("Warped Image 2")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("Stitched Result")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save results
        cv2.imwrite("warped_image2.jpg", warped_img2)
        cv2.imwrite("stitched_result.jpg", result)
        
    except Exception as e:
        print(f"Automatic stitching failed: {e}")
        print("Trying manual cropping method...")
        
        try:
            # Try different overlap percentages
            for overlap in [0.2, 0.3, 0.4]:
                try:
                    result = manual_stitch_with_crop(image1_path, image2_path, overlap)
                    plt.figure(figsize=(15, 10))
                    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                    plt.title(f"Manual Stitching (Overlap: {overlap*100}%)")
                    plt.axis('off')
                    plt.show()
                    
                    cv2.imwrite(f"manual_stitch_{int(overlap*100)}.jpg", result)
                    print(f"Manual stitching with {overlap*100}% overlap saved")
                    break
                except:
                    continue
                    
        except Exception as e2:
            print(f"All methods failed: {e2}")
            print("Please check your images and try again.")