import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import time

'''
Gif: https://tenor.com/view/fire-gif-24826882
'''

class ChangeBackground(object):
    
    def __init__(self):
        pass

    def load_gif(self, gif_path, gif, image_path)->any:
        im = Image.open(gif_path+'/'+gif[0])
        for frame in range(im.n_frames):
            im.seek(frame)
            im.save(image_path+'/'+str(frame)+"_img.png")

    def execute(self)->any:
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        gif_path =  'gifs'
        gif = os.listdir(gif_path)
        image_path = 'images'
        self.load_gif(gif_path, gif, image_path)
        images = os.listdir(image_path)
        image_index = 0
        velocidade_animacao = 0.08
        bg_image = cv2.imread(image_path+'/'+images[image_index])

        while cap.isOpened():
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            height, width, channe = frame.shape
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = selfie_segmentation.process(RGB)
            mask = results.segmentation_mask
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.6
            bg_image = cv2.resize(bg_image, (width, height))
            output_image = np.where(condition, frame, bg_image)
            cv2.putText(output_image, "Hold down D to disable", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
            cv2.imshow("Output", output_image)
            if image_index != len(images)-1:
                image_index += 1
            else:
                image_index = 0
            time.sleep(velocidade_animacao)
            bg_image = cv2.imread(image_path+'/'+images[image_index])
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('d'):
                bg_image = frame                
        cap.release()
        cv2.destroyWindow('Output')

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    changeBkg = ChangeBackground()
    changeBkg.execute()