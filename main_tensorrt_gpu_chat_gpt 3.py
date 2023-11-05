import random
import sys
from unittest import result
import torch
import pyautogui
import pygetwindow
import gc
import numpy as np
import cv2
import time
import win32api
import win32con
import pandas as pd
from utils.general import (cv2, non_max_suppression, xyxy2xywh)
from models.common import DetectMultiBackend
import dxcam
import cupy as cp
import math

def normalize_angle(angle):
    while angle <= -math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle

def smooth_angle(from_angle, to_angle, percent):
    vec_delta = [from_angle[0] - to_angle[0], from_angle[1] - to_angle[1]]
    vec_delta[0] = normalize_angle(vec_delta[0])
    vec_delta[1] = normalize_angle(vec_delta[1])
    vec_delta[0] *= percent
    vec_delta[1] *= percent
    to_angle[0] = from_angle[0] - vec_delta[0]
    to_angle[1] = from_angle[1] - vec_delta[1]


def load_model():
    model = DetectMultiBackend('yolov5s.engine', device=torch.device('cuda'), dnn=False, data='', fp16=True)
    return model

# Визуализация обнаруженных объектов с помощью bounding box

def plot_boxes(img, boxes):
    for box in boxes:
        x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img
def main():
    start_time = time.time()
    # Portion of screen to be captured (This forms a square/rectangle around the center of screen)
    screenShotHeight = 320
    screenShotWidth = 320

    some_threshold = 100
    # For use in games that are 3rd person and character model interferes with the autoaim
    # EXAMPLE: Fortnite and New World
    aaRightShift = 0

    # Autoaim mouse movement amplifier
    aaMovementAmp = .27

    # Aim smoothing factor (0 to 1, 0 means no smoothing, 1 means no movement)
    aim_smooth = 0.1
    aim_time = 1.1  # adjust value as needed

    # Person Class Confidence
    confidence = 0.4

    # What key to press to quit and shutdown the autoaim
    aaQuitKey = "P"

    # If you want to main slightly upwards towards the head
    headshot_mode = True

    # Displays the Corrections per second in the terminal
    cpsDisplay = True

    # Set to True if you want to get the visuals
    visuals = False

    antishake = True

    # Selecting the correct game window
    try:
        videoGameWindows = pygetwindow.getAllWindows()
        print("=== All Windows ===")
        videoGameWindow = None

        for index, window in enumerate(videoGameWindows):
            # Only output the window if it has a meaningful title
            if window.title != "":
                print("[{}]: {}".format(index, window.title))
                if "RedM™ by Cfx.re" in window.title or "Counter-Strike: Global Offensive" in window.title or "PUBG: BATTLEGROUNDS" in window.title:
                    videoGameWindow = window
                    break

        if videoGameWindow is None:
            try:
                userInput = int(input(
                    "Please enter the number corresponding to the window you'd like to select: "))
                videoGameWindow = videoGameWindows[userInput]
            except ValueError:
                print("You didn't enter a valid number. Please try again.")
                return
    except Exception as e:
        print("Failed to select game window: {}".format(e))
        return

    # Activate that Window
    activationRetries = 30
    activationSuccess = False
    while (activationRetries > 0):
        try:
            videoGameWindow.activate()
            activationSuccess = True
            break
        except pygetwindow.PyGetWindowException as we:
            print("Failed to activate game window: {}".format(str(we)))
            print("Trying again... (you should switch to the game now)")
        except Exception as e:
            print("Failed to activate game window: {}".format(str(e)))
            print("Read the relevant restrictions here: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setforegroundwindow")
            activationSuccess = False
            activationRetries = 0
            break
        # wait a little bit before the next try
        time.sleep(3.0)
        activationRetries = activationRetries - 1
    # if we failed to activate the window then we'll be unable to send input to it
    # so just exit the script now
    if activationSuccess == False:
        return
    print("Successfully activated the game window...")

    # Setting up the screen shots
    sctArea = {"mon": 1, "top": videoGameWindow.top + (videoGameWindow.height - screenShotHeight) // 2,
                         "left": aaRightShift + ((videoGameWindow.left + videoGameWindow.right) // 2) - (screenShotWidth // 2),
                         "width": screenShotWidth,
                         "height": screenShotHeight}

    # Starting screenshoting engine
    left = aaRightShift + \
        ((videoGameWindow.left + videoGameWindow.right) // 2) - (screenShotWidth // 2)
    top = videoGameWindow.top + \
        (videoGameWindow.height - screenShotHeight) // 2
    right, bottom = left + screenShotWidth, top + screenShotHeight

    region = (left, top, right, bottom)

    camera = dxcam.create(region=region)
    if camera is None:
        print("""DXCamera failed to initialize. Some common causes are:
        1. You are on a laptop with both an integrated GPU and discrete GPU. Go into Windows Graphic Settings, select python.exe and set it to Power Saving Mode.
         If that doesn't work, then read this: https://github.com/SerpentAI/D3DShot/wiki/Installation-Note:-Laptops
        2. The game is an exclusive full screen game. Set it to windowed mode.""")
        return
    camera.start(target_fps=160, video_mode=True)

    # Calculating the center Autoaim box
    cWidth = sctArea["width"] / 2
    cHeight = sctArea["height"] / 2

    # Used for forcing garbage collection
    count = 0
    sTime = time.time()

    # Loading yolov8n Small AI Model
    model = load_model()
    stride, names, pt = model.stride, model.names, model.pt

    # Used for colors drawn on bounding boxes
    COLORS = np.random.uniform(0, 255, size=(1500, 3))


    # Main loop Quit if Q is pressed
    last_mid_coord = None
    with torch.no_grad():
        while win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0:


            npImg = cp.array([camera.get_latest_frame()])
            im = npImg / 255
            im = im.astype(cp.half)

            im = cp.moveaxis(im, 3, 1)
            im = torch.from_numpy(cp.asnumpy(im)).to('cuda')

            # # Converting to numpy for visuals
            # im0 = im[0].permute(1, 2, 0) * 255
            # im0 = im0.cpu().numpy().astype(np.uint8)
            # # Image has to be in BGR for visualization
            # im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)

            # Detecting all the objects
            results = model(im)

            # Преобразование обратно в исходное изображение (в формате numpy)
            #orig_img = cp.asnumpy(npImg[0]).astype(np.uint8)
            #orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            #boxes = [result[:4] for result in results]

            # Нанесение bounding box на исходное изображение
            #boxed_img = plot_boxes(orig_img, boxes)

            # Filter the detections to only include the "person" class

            pred = non_max_suppression( results, confidence, confidence, 0, False, max_det=10)



            targets = []
            for i, det in enumerate(pred):
                s = ""
                gn = torch.tensor(im.shape)[[0, 0, 0, 0]]
                if len(det):
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}, "  # add to string

                    for *xyxy, conf, cls in reversed(det):
                        targets.append((xyxy2xywh(torch.tensor(xyxy).view(
                            1, 4)) / gn).view(-1).tolist() + [float(conf)])  # normalized xywh

            targets = pd.DataFrame( targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"])

            # If there are people in the center bounding box
            if len(targets) > 0:
                # Get the last persons mid coordinate if it exists
                if last_mid_coord:
                    center_x = screenShotWidth // 2
                    center_y = screenShotHeight // 2

                    targets['last_mid_x'] = last_mid_coord[0]
                    targets['last_mid_y'] = last_mid_coord[1]
                    # Take distance between current person mid coordinate and last person mid coordinate



                    if antishake:
                        targets['dist_to_center'] = np.linalg.norm( targets.loc[:, ['current_mid_x', 'current_mid_y']].values - [center_x, center_y], axis=1)
                        targets.sort_values(by="dist_to_center", ascending=True, inplace=True)
                    else:
                        targets['dist'] = np.linalg.norm(targets.iloc[:, [0, 1]].values - targets.iloc[:, [4, 5]], axis=1)
                        targets.sort_values(by="dist", ascending=False)
                # Take the first person that shows up in the dataframe (Recall that we sort based on Euclidean distance)
                xMid = targets.iloc[0].current_mid_x + aaRightShift
                yMid = targets.iloc[0].current_mid_y

                box_height = targets.iloc[0].height

                # Calculate aspect ratio and area of the object
                aspect_ratio = targets.iloc[0].width / targets.iloc[0].height
                area_ratio = (targets.iloc[0].width * targets.iloc[0].height) / (screenShotWidth * screenShotHeight)

                # Set thresholds for aspect ratio and area
                aspect_ratio_threshold = 0.6  # adjust value to determine a shape close to a square
                area_ratio_threshold = 0.4  # adjust value to determine an object occupying a large part of the frame

                # Check if the object's shape is close to a square and occupies a large part of the frame
                if aspect_ratio > aspect_ratio_threshold and area_ratio > area_ratio_threshold:
                    headshot_mode = False
                else:
                    headshot_mode = True

                if headshot_mode:
                    headshot_offset = box_height * 0.38
                else:
                    headshot_offset = box_height * 0.2

                # добавляем проверку на расположение цели на экране

                left = targets.iloc[0].current_mid_x
                top = targets.iloc[0].current_mid_y
                right = targets.iloc[0].current_mid_x + targets.iloc[0].width
                bottom = targets.iloc[0].current_mid_y + targets.iloc[0].height

                # Проверяем, находятся ли координаты центра экрана внутри рамки цели
                if last_mid_coord and not (left < cWidth < right and top < cHeight < bottom):
                    # Apply aim smoothing
                    print( "aim smoothing")
                    smooth_x = last_mid_coord[0] + (xMid - last_mid_coord[0]) * (1 - aim_smooth)
                    smooth_y = last_mid_coord[1] + (yMid - last_mid_coord[1]) * (1 - aim_smooth)

                else:
                    smooth_x = xMid
                    smooth_y = yMid

                mouseMove = [smooth_x - cWidth, (smooth_y - headshot_offset) - cHeight]

                if win32api.GetKeyState(0x06):
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(mouseMove[0] * aaMovementAmp),
                                         int(mouseMove[1] * aaMovementAmp), 0, 0)
                last_mid_coord = [smooth_x, smooth_y]



            else:
                last_mid_coord = None
                start_time = time.time()
                elapsed_time = 0

            # See what the bot sees
            if visuals:
                npImg = cp.asnumpy(npImg[0])
                # Loops over every item identified and draws a bounding box
                for i in range(0, len(targets)):
                    halfW = round(targets["width"][i] / 2)
                    halfH = round(targets["height"][i] / 2)
                    midX = targets['current_mid_x'][i]
                    midY = targets['current_mid_y'][i]
                    (startX, startY, endX, endY) = int(
                        midX + halfW), int(midY + halfH), int(midX - halfW), int(midY - halfH)

                    idx = 0
                    # draw the bounding box and label on the frame
                    label = "{}: {:.2f}%".format(
                        "person", targets["confidence"][i] * 100)
                    cv2.rectangle(npImg, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(npImg, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            # Forced garbage cleanup every second
            count += 1
            if (time.time() - sTime) > 1:
                if cpsDisplay:
                    print("CPS: {}".format(count))
                count = 0
                sTime = time.time()

            # Uncomment if you keep running into memory issues
            # gc.collect(generation=0)

            # See visually what the Aimbot sees
            if visuals:
                cv2.imshow('Live Feed', npImg)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    exit()
    camera.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        exc_type, exc_value, exc_traceback = sys.exc_info()  # Исправление здесь
        traceback.print_exception(exc_type, exc_value, exc_traceback)  # Исправление здесь

