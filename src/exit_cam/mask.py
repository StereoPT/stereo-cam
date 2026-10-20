import cv2 as cv

BLUR_SIZE = (21, 21)  # smooths out camera noise before comparing (must be odd)
KERNEL_SIZE = (11, 11)  # how big a speck or hole the clean-up removes


def grayscale_and_blur(img):
    return cv.GaussianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY), BLUR_SIZE, 0)


def make_kernel():
    """The shape the mask clean-up works with."""
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, KERNEL_SIZE)


def compute_mask(frame, gray_background, threshold, kernel):
    """A black-and-white mask: white where the frame differs from the
    background by more than threshold, cleaned of specks and small holes.
    gray_background must come from grayscale_and_blur."""
    diff = cv.absdiff(grayscale_and_blur(frame), gray_background)
    _, mask = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
