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
        K = 5

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
        sensitivity = False

        for (x, y, w, h) in faces:
            detected = True

            # bounding rectangle for face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            print(x, y, w, h)
            # Shirt coordinates
            offset_y_bottom = 70
            offset_x_left = 30
            offset_x_right = 30
            shirt_y = y + h + offset_y_bottom
            shirt_w = w // 3
            shirt_h = w // 3

            try:
                # shirt region 1 (left)
                shirt_x = x + (w // 3) - offset_x_left
                sensitivity_1 = self.shirt_region(blur, shirt_x, shirt_y, shirt_h, shirt_w)
                cv2.rectangle(img, (shirt_x, shirt_y),
                              (shirt_x+shirt_w, shirt_y+shirt_h), (255, 0, 0), 2)

                # shirt region 2 (right)
                shirt_x = x + (w // 3) + offset_x_right
                sensitivity_2 = self.shirt_region(blur, shirt_x, shirt_y, shirt_h, shirt_w)
                cv2.rectangle(img, (shirt_x, shirt_y),
                              (shirt_x+shirt_w, shirt_y+shirt_h), (255, 0, 0), 2)

                # shirt region 3 (bottom)
                shirt_y = y + h + 130
                shirt_x = x + (w // 3)
                sensitivity_3 = self.shirt_region(blur, shirt_x, shirt_y, shirt_h, shirt_w)
                cv2.rectangle(img, (shirt_x, shirt_y),
                              (shirt_x+shirt_w, shirt_y+shirt_h), (255, 0, 0), 2)

                sensitivity = sensitivity_1 or sensitivity_2 or sensitivity_3
                break
            except cv2.error:
                logging.debug(f"Error in K means {self.img_path[3:]}")
                continue
            except ZeroDivisionError:
                logging.debug(f"Division by Zero {self.img_path[3:]}")
                continue

        if detected:
            self.img_path = self.img_path.split('/')[-1]

            cv2.imwrite(('uniform/' if sensitivity else 'no_uniform/')+self.img_path, img)
        else:
            # cv2.imwrite('noface/shirt_'+f[3:], img)
            logging.debug(f"No face detected in {self.img_path[3:]}")

    def shirt_region(self, blur, shirt_x, shirt_y, shirt_h, shirt_w, name):
        crop_img = blur[shirt_y:shirt_y+shirt_h, shirt_x:shirt_x+shirt_w]
        crop_img = self.compress(crop_img)

        avg_color_per_row = np.average(crop_img, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)

        r, g, b = map(lambda x: int(x) if not math.isnan(x) else x,
                      [avg_color[2], avg_color[1], avg_color[0]])

        cv2.imwrite('crop/'+name+self.img_path.split('/')[-1], crop_img)

        return self.check_uniform(r, g, b)

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

        return True if diff <= 20 and max(g, b) - 20 >= r else False


if __name__ == '__main__':
    logging.basicConfig(filename='/var/tmp/shirt.log',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG)
    files = glob('../*.jpg')

    for f in files:
        shatsu_obj = Shatsu(f)
        shatsu_obj.detect()
