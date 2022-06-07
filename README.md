# pose-classification

## Requirements

- Binaries for running openpose: either from a working installation 
or downloaded from [here](https://drive.google.com/file/d/1fX5Qw3IA93OovFMhNEPR9bnY3HHMfyXY/view?usp=sharing) (should work with an Nvidia GPU);
- Body models for openpose ([BODY_25](https://drive.google.com/file/d/1azo49hjKNc2U47uilI7PQtT4R3fGRUZ4/view?usp=sharing) or COCO or MPI).

Both should be placed in bin/ and models/ folders inside the project folder. 

## Usage

Run the pose_classification.py script via an IDE  or command-line.
The script requires paths to image directories as command-line arguments.

(e.g. python pose_classification.py --image-dirs ./media/static_images ./media/dynamic_images)
