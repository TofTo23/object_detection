import cv2
import numpy as np
from pathlib import Path

# range of colors in hsv
lower_green = np.array([30, 50, 50])
upper_green = np.array([90, 235, 235])

lower_red = np.array([0, 170, 170])
upper_red = np.array([10, 235, 235])


def colorRange(img, low, high):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low, high)

    pixel = hsv[mask > 0]

    lowHue = np.percentile(pixel[:, 0], 10)
    lowSaturation = np.percentile(pixel[:, 1], 10)
    lowValue = np.percentile(pixel[:, 2], 10)
    
    highHue = np.percentile(pixel[:, 0], 90)
    highSaturation = np.percentile(pixel[:, 1], 90)
    highValue = np.percentile(pixel[:, 2], 90)

    lowRange = np.array([int(lowHue), int(lowSaturation), int(lowValue)])
    highRange = np.array([int(highHue), int(highSaturation), int(highValue)])

    return lowRange, highRange


def loadImages():
    path = Path()
    images_path = list(path.glob('*/*.jpg'))
    template_path = next(path.glob('*.png'), None)

    images = []
    images_names = []
    template = cv2.imread(str(template_path))

    if template is None:
        print('Nie wczytano szablonu z ', template_path)

    for idx, img_path in enumerate(images_path):
        images.append(cv2.imread(str(img_path)))
        images_names.append(str(img_path.name))
        if images[idx] is None:
            print("Nie wczytano obrazu z ", img_path)

    return images, images_names, template


def selectColor(picture, lower_range, upper_range):
    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    result = cv2.bitwise_and(picture, picture, mask=mask)
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    return result_gray


def matchGreenRed(picture, grinchGreenLow, grinchGreenUp, grinchRedLow, grinchRedUp):
    width = 30
    height = 30
    kernel = np.ones((30, 30), np.uint8)

    maskGreen = selectColor(picture, grinchGreenLow, grinchGreenUp)
    maskRed = selectColor(picture, grinchRedLow, grinchRedUp)

    countGreen = cv2.boxFilter(maskGreen, -1, (height, width), normalize=False)
    countRed = cv2.boxFilter(maskRed, -1, (height, width), normalize=False)

    countBoth = cv2.bitwise_and(countRed, countGreen)
    closing = cv2.morphologyEx(countBoth, cv2.MORPH_CLOSE, kernel)
    _, binary = cv2.threshold(closing, 5, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)

    return stats, num_labels

def scalePicture(picture):
    size = picture.shape
    y_n = 900 / size[0]
    x_n = 1440 / size[1]

    scale = min(x_n, y_n)
    picture = cv2.resize(picture, (0,0), fx=scale, fy=scale)

    return picture


def compCorners(stat):
    area = np.array(stat[:, cv2.CC_STAT_AREA])
    left = np.array(stat[:, cv2.CC_STAT_LEFT])
    top = np.array(stat[:, cv2.CC_STAT_TOP])
    width = np.array(stat[:, cv2.CC_STAT_WIDTH])
    height = np.array(stat[:, cv2.CC_STAT_HEIGHT])

    idx_area = area.argsort()[-3:]
    area = area[idx_area]
    left = left[idx_area]
    top = top[idx_area]
    width = width[idx_area]
    height = height[idx_area]
    
    left_top = (left, top)
    right_bottom = (left + width, top + height)

    
    return left_top, right_bottom


if __name__ == "__main__":
    images, images_names, template = loadImages()

    grinchRedLow, grinchRedUp = colorRange(template, lower_red, upper_red)
    grinchGreenLow, grinchGreenUp = colorRange(template, lower_green, upper_green)

    for idx, img in enumerate(images):
        stat, num_labels = matchGreenRed(img, grinchGreenLow, grinchGreenUp, grinchRedLow, grinchRedUp)
        if num_labels > 2:
            left_top, right_bottom = compCorners(stat)

            img = cv2.rectangle(img, (left_top[0][0], left_top[1][0]), (right_bottom[0][0], right_bottom[1][0]), 220, 3)
            img = cv2.rectangle(img, (left_top[0][1], left_top[1][1]), (right_bottom[0][1], right_bottom[1][1]), 220, 3)

            img = scalePicture(img)
            
            cv2.imshow(images_names[idx], img)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()