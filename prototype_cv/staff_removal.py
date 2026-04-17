import cv2
import numpy as np

def extract_staff_lines(image_path):
    """
    Reads an image, binarizes it, and extracts horizontal staff lines.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None, None, None
    
    # Binarize (invert so text/lines are white, background is black)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Create horizontal kernel to detect long horizontal lines (Main staff lines)
    kernel_len = img.shape[1] // 30
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    
    # Use morphology to isolate main staff lines
    staff_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # To detect short ledger lines (加线) which connect notes above/below the staff
    short_kernel_len = img.shape[1] // 100
    short_horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (short_kernel_len, 1))
    all_horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, short_horizontal_kernel, iterations=1)
    
    # We only want thin lines (ledger lines), so we remove thick horizontal components (like beams)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    thick_horiz_lines = cv2.morphologyEx(all_horiz_lines, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    ledger_lines = cv2.subtract(all_horiz_lines, thick_horiz_lines)
    
    # Combine main staff lines and ledger lines
    all_lines_to_remove = cv2.add(staff_lines, ledger_lines)
    
    # Subtract all lines from the binary image
    music_symbols = cv2.subtract(binary, all_lines_to_remove)
    
    return all_lines_to_remove, music_symbols, binary

if __name__ == "__main__":
    # Test script will be here
    pass
