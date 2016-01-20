# Photo-a-day Aligner 

A few tools to help with daily self-portrait projects:

* `align.py`: Take a set of photo-a-day images, and align them based on the
  detected face. Also perform RGB scaling so that all the faces have the same
  average RGB value. Also outputs an image `mask.png` which is used by the next
  script. Duplicate images, images with no face, and images with more than one
  face are dropped at this stage.
* `framedrop.py`: Produce a file list, based on the output files of the above
  script. The output will have approximately `(100 / N)` % of the input images
  (`N` is `10` by default). Output frames are selected to avoid temporal
  discontinuities in the face area.
* `make_vid.sh`: A shell script which calls `mplayer` to encode the file list
  produced by the above into a .h264 MP4 file.

The Python scripts are currently configured by changing the capital letter
variables defind at the top of each file. (This may change in the future).
Similarly, video compression parameters are changed by editing `make_vid.sh`.

## Requirements

`align.py` requires `numpy`, `dlib`, `scipy`, and `cv2`.
`framedrop.py` requires `cv2` and `numpy`.
`make_vid.sh` requires `mplayer` and suitable codecs to be installed.

