import matplotlib.pyplot as plt
from mtcnn import MTCNN
import cv2
import cvlib as cv
import numpy as np
import math

class FaceAligner:
    """
    Adapted from:
    https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
    """
    def __init__(self, zoom_out_factor=0.2,
        desiredFaceWidth=256, desiredFaceHeight=None):
        """
        Construct a face aligner.
        :param zoom_out_factor: Increase to include more background
        :param desiredFaceWidth: The desired image width
        :param desiredFaceHeight: The desired image height
        """
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = MTCNN()
        self.desiredLeftEye = (zoom_out_factor, 0.2)
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, shift=50, use_alternative=False, use_gpu=True, visualise_process=False):
        """
        Find a face in an image and horizontally align the eyes
        :param image: The image to extract the face from
        :param shift: The amount to vertically shift the face region. The bigger the number, the closer to the chin
        :param use_alternative: If the MTCNN face detection fails, you can try cvlib's
        :param use_gpu: True to use GPU with alternative detection.
        :return: a face crop with horizontally-aligned eyes
        """
        # test the image shape
        if len(image.shape) != 3:
            raise TypeError("Align function expects an image shape of (height, width, channels)")

        vid_height, vid_width, channels = image.shape
        # convert the landmark (x, y)-coordinates to a NumPy array
        face_dic = self.predictor.detect_faces(image)
        if face_dic is None or len(face_dic) == 0:
            if use_alternative == False:
                # uncomment these lines if you wish to see why the face could not be detection
                # plt.imshow(image)
                # plt.show()
                return None
            else:
                # use alternate face detector
                faces, confidences = cv.detect_face(image, use_gpu=use_gpu)

                if confidences is not None:
                    if len(confidences) > 0:
                        for face, confidence in zip(faces, confidences):
                            (startX, startY) = face[0], face[1]
                            (endX, endY) = face[2], face[3]
                            width = endX - startX
                            height = endY - startY
                            inset = (height - width) / 2
                            # if the coordinates exceeds the image, return
                            if startX < 0 or endX > vid_width or startY < 0 or endY > vid_height or width > vid_width or height > vid_height:
                                continue
                            # draw bounding box
                            startY = startY + int(math.ceil(inset)) + shift
                            endY = endY - int(math.floor(inset)) + shift
                            new_height = endY - startY
                            crop_image = image[startY:endY, startX:endX]


                            if new_height != width:
                                raise Exception("Height and width not equal")
                            return crop_image
        else:
            if visualise_process:
                frame_list = []
            # the faces were found
            # we use the first face
            dic = face_dic[0]

            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = dic['keypoints']['left_eye']
            (rStart, rEnd) = dic['keypoints']['right_eye']
            if visualise_process:
                detection_img = image.copy()
                frame_list.append(image)
                detection_img = cv2.line(detection_img, (lStart, lEnd), (rStart, rEnd), (0, 0, 255), 4)
                cv2.circle(detection_img, (lStart, lEnd), 2, (255,0,0), 4,)
                cv2.circle(detection_img, (rStart, rEnd), 2, (255,0,0), 4)
                detection_img = cv2.rectangle(detection_img, (dic['box'][0], dic['box'][1]), (dic['box'][0]+dic['box'][2], dic['box'][1]+dic['box'][3]), (0, 255,0), 2)

                alpha = 1.0
                overlay = cv2.addWeighted(detection_img, alpha, image, 1 - alpha, 0)
                # plt.imshow(overlay)
                # plt.show()
                frame_list.append(overlay)


            # compute the angle between the eye centroids
            dY = lEnd - rEnd
            dX = lStart - rStart
            angle = np.degrees(np.arctan2(dY, dX)) - 180

            # compute the desired right eye x-coordinate based on the
            # desired x-coordinate of the left eye
            desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
            # determine the scale of the new resulting image by taking
            # the ratio of the distance between eyes in the *current*
            # image to the ratio of distance between eyes in the
            # *desired* image
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
            self.desiredFaceWidth = dic['box'][2]
            desiredDist *= self.desiredFaceWidth
            scale = desiredDist / dist
            # compute center (x, y)-coordinates (i.e., the median point)
            # between the two eyes in the input image
            eyesCenter = ((lStart + rStart) // 2, (lEnd + rEnd) // 2)
            # grab the rotation matrix for rotating and scaling the face
            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
            # update the translation component of the matrix
            tX = self.desiredFaceWidth * 0.5
            tY = self.desiredFaceWidth * 0.5 -shift#self.desiredFaceHeight * self.desiredLeftEye[1]
            M[0, 2] += (tX - eyesCenter[0])
            M[1, 2] += (tY - eyesCenter[1])
            # apply the affine transformation
            (w, h) = (self.desiredFaceWidth, self.desiredFaceWidth)
            output = cv2.warpAffine(image, M, (w, h),
                flags=cv2.INTER_CUBIC)
            if visualise_process:
                overlay = cv2.warpAffine(overlay, M, (w, h),
                    flags=cv2.INTER_CUBIC)
                # plt.imshow(overlay)
                # plt.show()
                frame_list.append(overlay)
                frame_list.append(output)
                return frame_list
            # return the aligned face
            return output