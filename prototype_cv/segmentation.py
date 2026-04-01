import cv2
import numpy as np

def extract_symbols(music_symbols_img):
    """
    Takes an image with staff lines removed (mostly music symbols)
    and segments out individual connected components.
    """
    # Morphological operations to clean up fragments left after staff line removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Close small gaps
    cleaned = cv2.morphologyEx(music_symbols_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Let's remove the MORPH_OPEN step entirely, because it might be eroding away 
    # very thin stems or small dots (like staccato or repeat dots)
    # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find connected components (with stats)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    
    symbols = []
    # Loop over components (excluding background label 0)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter out very small blobs (noise) or huge blocks (margins/logos)
        # We need to be careful with the area thresholds, smaller symbols like dots or thin stems might be filtered out
        # Let's lower the minimum area threshold and increase the max to ensure we catch everything
        if area > 5 and area < 20000:
            symbols.append((x, y, w, h, area))
            
    # Sort symbols by x-coordinate (left to right)
    symbols.sort(key=lambda s: s[0])
    
    return symbols, cleaned

if __name__ == "__main__":
    pass