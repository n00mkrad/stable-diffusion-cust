import cv2

def preprocess(image, pyrUp_iters = 3):
    H, W, C = image.shape
    detected_map = cv2.resize(image, (W // (2 ** pyrUp_iters), H // (2 ** pyrUp_iters)), interpolation=cv2.INTER_AREA) 
    for _ in range(pyrUp_iters):
        detected_map = cv2.pyrUp(detected_map)
    return detected_map