#!/usr/bin/env python

import cv2
import mediapipe as mp

class handDetector():
    def __init__(self):

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, hand="first", draw=True):
        self.lmList =[]
        num_of_hands = 0

        if hand == "first":
            hand_number = -1

        else:
            hand_number = 0

        if self.results.multi_hand_landmarks:
            num_of_hands = len(self.results.multi_hand_landmarks)
            main_hand = self.results.multi_hand_landmarks[hand_number]

            for id, lm in enumerate(main_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])

            if draw:
                self.mpDraw.draw_landmarks(img, main_hand, self.mpHands.HAND_CONNECTIONS)

        return num_of_hands, self.lmList

