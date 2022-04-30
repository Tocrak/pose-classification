import sys
import cv2
import os
from sys import platform

dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        sys.path.append(dir_path + '/bin')
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/bin;'
        import pyopenpose as op
    else:
        sys.path.append('/bin')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu),
        # you can also access the OpenPose/python module from there. This will install OpenPose and the python
        # library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose_example import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found.')
    raise e


class OpenposeRunner:
    def __init__(self, model_folder="./models/", display=False):
        self.op_wrapper = op.WrapperPython()
        self.params = dict()
        self.params["model_folder"] = model_folder
        self.display = display
        self.keypoints = list()

    def start_openpose(self):
        self.op_wrapper.configure(self.params)
        self.op_wrapper.start()

    def process_images(self, image_paths):
        for imagePath in image_paths:
            datum = op.Datum()
            image_to_process = cv2.imread(imagePath)
            datum.cvInputData = image_to_process
            self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
            self.keypoints.extend(datum.poseKeypoints)

            if self.display:
                self.display_images(datum)

    def display_images(self, datum):
        cv2.imshow("OpenPose 1.7.0 - Detect Keypoints", datum.cvOutputData)

    def get_image_paths(self, image_dir):
        return op.get_images_on_directory(image_dir)

    def run(self, image_dir="./media/", image_path=None):
        self.keypoints = list()
        self.start_openpose()

        if image_path:
            self.process_images([image_path])
        else:
            image_paths = self.get_image_paths(image_dir)
            self.process_images(image_paths)
