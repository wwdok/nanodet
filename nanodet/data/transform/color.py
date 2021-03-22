import numpy as np
import cv2
import random


def random_brightness(img, delta):
    img += random.uniform(-delta, delta)  # delta越大，代表这张图片随机生成的亮度范围越大
    return img


def random_contrast(img, alpha_low, alpha_up):
    img *= random.uniform(alpha_low, alpha_up)
    return img


def random_saturation(img, alpha_low, alpha_up):
    hsv_img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    hsv_img[..., 1] *= random.uniform(alpha_low, alpha_up)
    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return img


def normalize(meta, mean, std):
    img = meta['img'].astype(np.float32)
    mean = np.array(mean, dtype=np.float64).reshape(1, -1)
    stdinv = 1 / np.array(std, dtype=np.float64).reshape(1, -1)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    meta['img'] = img
    return meta


def _normalize(img, mean, std):
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    img = (img - mean) / std
    return img


def color_aug_and_norm(meta, kwargs):
    img = meta['img'].astype(np.float32) / 255

    if 'brightness' in kwargs and random.randint(0, 1):
        img = random_brightness(img, kwargs['brightness'])

    if 'contrast' in kwargs and random.randint(0, 1):
        img = random_contrast(img, *kwargs['contrast'])

    if 'saturation' in kwargs and random.randint(0, 1):
        img = random_saturation(img, *kwargs['saturation'])
    # cv2.imshow('trans', img)
    # cv2.waitKey(0)
    img = _normalize(img, *kwargs['normalize'])
    meta['img'] = img
    return meta


if __name__ == '__main__':
    img = cv2.imread(r'../../../demo/input/Snipaste_2021-03-13_11-04-57.jpg').astype(np.float32) / 255

    # img1 = random_brightness(img, 0.5)
    # cv2.imshow('random_brightness : 0.5', img1)

    # img2 = random_contrast(img, 0.5, 2.0)
    # cv2.imshow('random_contrast : 0.5 - 2.0', img2)

    # img3 = random_saturation(img, 0.5, 1.5)
    # cv2.imshow('random_contrast : 0.5 - 1.5', img3)

    mean = [103.53, 116.28, 123.675]
    std = [57.375, 57.12, 58.395]
    img4 = _normalize(img, mean, std)
    cv2.imshow('_normalize : 115 - 57', img4)

    cv2.waitKey(0)
