import cv2
import numpy as np
import pytesseract

def get_text_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check if flowchart
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 800:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 30 and h > 15 and 0.2 < w/h < 10:
                boxes.append((x, y, w, h))
    
    # If flowchart (3+ boxes), extract from boxes
    if len(boxes) >= 3:
        texts = []
        for x, y, w, h in boxes:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (w*2, h*2))
            text = pytesseract.image_to_string(roi, config='--oem 3 --psm 6').strip()
            if text:
                texts.append(text)
        return '\n'.join(texts)
    
    # Regular document processing
    enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    return pytesseract.image_to_string(enhanced, config='--oem 3 --psm 6').strip()

# Usage
text = get_text_from_image("/Users/dheerajnagpal/Projects/llm-knowledge-graph-construction/llm-knowledge-graph/IMG_4346.jpg")
print(text)