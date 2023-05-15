import numpy as np
import cv2 as cv
import onnxruntime
class FRCN:
    def __init__(self, modelPath, backendId=0, targetId=0):
        self._model_path = modelPath

        self._backendId = backendId
        self._targetId = targetId

        # self._model = cv.dnn.readNet(self._model_path)
        self._model = onnxruntime.InferenceSession(self._model_path)
        # self._model.setPreferableBackend(self._backendId)
        # self._model.setPreferableTarget(self._targetId)


        # self._inputSize = [100, 32] # Fixed


        # self._targetVertices = np.array([
        #     [0, self._inputSize[1] - 1],
        #     [0, 0],
        #     [self._inputSize[0] - 1, 0],
        #     [self._inputSize[0] - 1, self._inputSize[1] - 1]
        # ], dtype=np.float32)

    @property
    def name(self):
        return self.__class__.__name__

    '''
        def setBackend(self, backend_id):
            self._backendId = backend_id
            self._model.setPreferableBackend(self._backendId)

        def setTarget(self, target_id):
            self._targetId = target_id
            self._model.setPreferableTarget(self._targetId)
        '''
    def _preprocess(self, image):

        input_img = image.astype(np.float32)

        # HWC to NCHW
        input_img = np.transpose(input_img, [2, 0, 1])
        input_img = np.expand_dims(input_img, 0)


        return input_img


    def infer(self, image):
        # Preprocess
        input_img = self._preprocess(image)




        # onnxruntime
        ort_inputs = {'input': input_img}
        outputBlob = self._model.run(['output'], ort_inputs)[0]



        # Postprocess
        results = self._postprocess(outputBlob)


        return results

    def _postprocess(self, ort_output):


        ort_output = np.squeeze(ort_output, 0)
        ort_output = np.clip(ort_output, 0, 255)
        ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)

        return ort_output