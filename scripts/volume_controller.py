#!/usr/bin/env python3

import cv2
from math import hypot
import alsaaudio
import numpy as np
import hand_detector as detector


class VolumeContoller():
    def __init__(self, cap):
        self.detector = detector.handDetector()
        self.volume = alsaaudio.Mixer()

        self.volMin, self.volMax = self.volume.getrange(units=0)
        self.volRange = self.volume.getrange(units=0)
        self.cap = cap


    def run(self):
        self.cap = cv2.VideoCapture(0)
        volPer = 0
        volBar = 350
        count = 0

        while True:
            success, img = cap.read()
            img = self.detector.findHands(img)
            numOfHands, lmList = self.detector.findPosition(img, draw=True, hand="first")
            if len(lmList) != 0:

                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                length = hypot(x2 - x1, y2 - y1)
                # print(length)
                vol = np.interp(length, [25, 270], [self.volMin, self.volMax])
                volBar = np.interp(length, [25, 270], [400, 150])
                volPer = np.interp(length, [25, 270], [0, 100])
                # print(vol)
                current_volume = self.volume.getvolume()
                self.volume.setvolume(vol + current_volume, None)

                if length < 50:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

                if numOfHands > 1:
                    count = count + 1

                    if count > 10:
                        break

                else:
                    count = 0

            cv2.putText(img, f"{int(volPer)} %", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)


            cv2.imshow("Image", img)
            cv2.waitKey(1)


            # success, img = cap.read()
            # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # results = self.hands.process(imgRGB)
            #
            # lmList = []
            # cx, cy = 0, 0
            #
            # if results.multi_hand_landmarks:
            #     for handlandmark in results.multi_hand_landmarks:
            #         for id, lm in enumerate(handlandmark.landmark):
            #             h, w, c = img.shape
            #             cx, cy = int(lm.x * w), int(lm.y * h)
            #             lmList.append([id, cx, cy])
            #         self.mpDraw.draw_landmarks(img, handlandmark, self.mpHands.HAND_CONNECTIONS)
            #
            #     if len(results.multi_hand_landmarks) > 1:
            #         count = count + 1
            #
            #         if count > 10:
            #             break
            #
            #     else:
            #         count = 0
            #
            # if lmList != []:
            #
            #     x1, y1 = lmList[4][1], lmList[4][2]
            #     x2, y2 = lmList[8][1], lmList[8][2]
            #
            #     cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            #     cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            #     cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
            #
            #     length = hypot(x2 - x1, y2 - y1)
            #
            #     if length < 30:
            #         cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)
            #
            #     vol = np.interp(length, [15, 250], [self.volMin, self.volMax])
            #     volBar = np.interp(length, [25, 250], [350, 150])
            #     volPer = np.interp(length, [25, 250], [0, 100])
            #
            #     self.volume.SetMasterVolumeLevel(vol, None)
            #
            #     # Hand range 15 - 220
            #     # Volume range -63.5 - 0.0
            #
            # cv2.rectangle(img, (50, 150), (75, 350), (0, 255, 0), 2)
            # cv2.rectangle(img, (50, int(volBar)), (75, 350), (0, 255, 0), cv2.FILLED)
            # cv2.putText(img, f'{int(volPer)} %', (44, 200), cv2.FONT_HERSHEY_COMPLEX, 0.6, (200, 0, 0), 2)
            #
            # cv2.imshow('Image', img)
            # if cv2.waitKey(1) & 0xff == ord('q'):
            #     break



cap = cv2.VideoCapture(0)
volume_controller = VolumeContoller(cap)
volume_controller.run()