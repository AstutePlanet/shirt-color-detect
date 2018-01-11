from glob import glob
import math
import logging

import numpy as np
import cv2


class Shatsu(object):
    def __init__(self, img_path):
        self.img_path = img_path
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    def compress(self, img):
        Z = img.reshape((-1, 3))

        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8

        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10,
                                        cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        return res2

    def detect(self):
        img = cv2.imread(self.img_path)

        blur = cv2.GaussianBlur(img, (7, 7), 0)
        # hsv_img = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
        # cv2.imwrite('hsv/shirt_'+f[3:], hsv_img)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            logging.debug(f"Error on gray scale conversion {self.img_path[3:]}")

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        detected = False

        for (x, y, w, h) in faces:
            detected = True

            # bounding rectangle for face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            print(x, y, w, h)
            # Shirt coordinates
            offset_y_bottom = 70
            offset_x_left = 30
            shirt_x = x + (w // 3) - offset_x_left
            shirt_y = y + h + offset_y_bottom
            shirt_w = w // 3
            shirt_h = w // 3

            print(shirt_x, shirt_y, shirt_w, shirt_h)

            crop_img = blur[shirt_y:shirt_y+shirt_h, shirt_x:shirt_x+shirt_w]
            try:
                crop_img = self.compress(crop_img)
            except cv2.error:
                logging.debug(f"Error in K means {self.img_path[3:]}")
                continue

            try:
                avg_color_per_row = np.average(crop_img, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)
            except ZeroDivisionError:
                logging.debug(f"Division by Zero {self.img_path[3:]}")
                continue

            font = cv2.FONT_HERSHEY_SIMPLEX
            r, g, b = map(lambda x: int(x) if not math.isnan(x) else x,
                          [avg_color[2], avg_color[1], avg_color[0]])
            text = f'{r}, {g}, {b}'
            # cv2.putText(img, check_uniform(r, g, b), (100, 100),
            #            font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, 'con: ' + self.color_detect(crop_img)+'%' + self.check_uniform(r,g,b), (50, 50),
                        font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            print(avg_color)

            # bounding rectangle for shirt
            cv2.rectangle(img, (shirt_x, shirt_y),
                          (shirt_x+shirt_w, shirt_y+shirt_h), (255, 0, 0), 2)
            break

        if detected:
            cv2.imwrite('new/shirt_'+self.img_path[3:], img)
            cv2.imwrite('crop/shirt_'+self.img_path[3:], crop_img)
        else:
            # cv2.imwrite('noface/shirt_'+f[3:], img)
            logging.debug(f"No face detected in {self.img_path[3:]}")

    def color_detect(self, img):
        confidence = 0
        for row in img:
            for b, g, r in row:
                diff = abs(g-b)
                if diff <= 20 and max(g, b) - 20 >= r:
                    confidence += 1
        return str(int((confidence / (img.shape[0]*img.shape[1])) * 100))

    def check_uniform(self, *rgb):
        r, g, b = rgb
        diff = abs(g-b)
        if diff <= 20 and max(g, b) - 20 >= r:
            return " in uniform"
        else:
            return " not in uniform"


if __name__ == '__main__':
    logging.basicConfig(filename='/var/tmp/shirt.log',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG)
    files = glob('../*.jpg')

    for f in files:
        shatsu_obj = Shatsu(f)
        shatsu_obj.detect()
