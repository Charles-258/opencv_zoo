
import sys
import argparse

import numpy as np
import cv2 as cv

from frcn_onnx import FRCN

# Check OpenCV version
assert cv.__version__ >= "4.7.0", \
       "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(
    description="An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (https://arxiv.org/abs/1507.05717)")
# parser.add_argument('--input', '-i', type=str, default="/Users/yiyaowang/Downloads/undergraduate/graduate_project/opencv_zoo/models/face_hallucination/demo/demo1.png",
#                     help='Usage: Set path to the input image. Omit for using default camera.')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='frcn.onnx',
                    help='Usage: Set model path, defaults to frcn.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--width', type=int, default=256,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
parser.add_argument('--height', type=int, default=256,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
parser.add_argument('--save', '-s', action='store_true', default=0,
                    help='Usage: Specify to save a file with results. Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true', default=1,
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()



if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    recognizer = FRCN(modelPath=args.model, backendId=backend_id, targetId=target_id)

    # If input is an image
    if args.input is not None:
        original_image = cv.imread(args.input)

        original_w = original_image.shape[1]
        original_h = original_image.shape[0]
        # scaleHeight = original_h / args.height
        # scaleWidth = original_w / args.width
        image = cv.resize(original_image, [args.width, args.height])

        # Inference
        results = recognizer.infer(image)

        # Save results if save is true
        if args.save:
            print('Resutls saved to result.jpg\n')
            cv.imwrite('result.png', results)

        # Visualize results in a new window
        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, results)
            cv.waitKey(0)


    else: # Omit input to call default camera
        deviceId = 0
        cap = cv.VideoCapture(deviceId)

        tm = cv.TickMeter()
        # while cv.waitKey(1) < 0:
        while cv.waitKey(1) < 0:
            hasFrame, original_image = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            original_w = original_image.shape[1]
            original_h = original_image.shape[0]
            # scaleHeight = original_h / args.height
            # scaleWidth = original_w / args.width
            frame = cv.resize(original_image, [args.width, args.height])

            # Inference
            tm.start()
            results = recognizer.infer(frame)  # results is a tuple
            tm.stop()

            # Visualize results in a new Window
            cv.imshow('orginal_pic', frame)
            cv.imshow('{} Demo'.format(recognizer.name), results)

            tm.reset()



