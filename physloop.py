import cv2
import numpy as np


def apply_physloop(img, blur_factor=2, center_frac=0.5):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // blur_factor), max(1, h // blur_factor)),
                       interpolation=cv2.INTER_AREA)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    m_y = int(h * (1 - center_frac) / 2)
    m_x = int(w * (1 - center_frac) / 2)
    bg[m_y:h - m_y, m_x:w - m_x] = img[m_y:h - m_y, m_x:w - m_x]
    return bg
