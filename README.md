# Photo-a-day Aligner 

A tools to help with daily self-portrait projects:


* `pada.py` which has a couple of sub-commands:
 * `align`: Take a set of photo-a-day images, and align them based on the
   detected face, and perform RGB scaling so that all the faces have the same
   average RGB value. Also outputs an image `mask.png` which is used by the
   next script. Duplicate images, images with no face, and images with more than
   one face are dropped at this stage.
 * `framedrop`: Produce a file list, based on the output files of the above
   script. The output will have approximately `(100 / N)` % of the input images
   (`N` is `10` by default). Output frames are selected to avoid temporal
   discontinuities in the face area.
* `make_vid.sh`: A shell script which calls `mencoder` to encode the file list
  produced by the above into a .h264 MP4 file.

See below for usage details.

## Recommended workflow

1. Create a directory for your project.

2. Copy `examples/pada.conf` into it. Change `predictor_path` to point to your
   dlib landmarks, [downloadable from here](http://sourceforge.net/projects/
dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2).

3. Create a sub-directory `input`, and place your input frames into it. When
   lexicographically sorted the file names should be in the correct order.

4. Run `pada.py align` to align and colour correct your input frames. At this
   point you can inspect the output in `./aligned`. If the results are not
   satisfactory change settings and repeat this step.

5. Run `pada.py framedrop` to select a sequence of good frames and output them
   to `filtered.txt`.

6. Run `make_vid.sh` to convert the above file list into a video, `output.mp4`.

## Usage

General `pada.py` options:

    $ pada.py --help
    usage: pada.py [-h] [--debug] [--config CONFIG] [--aligned-path ALIGNED_PATH]
                   [--aligned-extension ALIGNED_EXTENSION]
                   [--predictor-path PREDICTOR_PATH]
                   [--filtered-files FILTERED_FILES]
                   {print_config_paths,align,framedrop} ...

    positional arguments:
      {print_config_paths,align,framedrop}
                            Sub-command help
        print_config_paths  print config paths and exit
        align               align a set of images
        framedrop           Drop frames from a set of images

    optional arguments:
      -h, --help            show this help message and exit
      --debug               Print debug information
      --config CONFIG       Config file path
      --aligned-path ALIGNED_PATH
                            Path where aligned images will be stored
      --aligned-extension ALIGNED_EXTENSION
                            Extension (and filetype) to use for aligned images.
      --predictor-path PREDICTOR_PATH
                            DLib face predictor dat file
      --filtered-files FILTERED_FILES
                            File to write filtered files to

`pada.py align` options:

    $ pada.py align --help
    usage: pada.py align [-h] [--input-glob INPUT_GLOB] [--img-thresh IMG_THRESH]

    optional arguments:
      -h, --help            show this help message and exit
      --input-glob INPUT_GLOB
                            Input files glob
      --img-thresh IMG_THRESH
                            Max duplicate frame delta

`pada.py framedrop` options:

    $ pada.py framedrop --help
    usage: pada.py framedrop [-h] [--erode-amount ERODE_AMOUNT]
                             [--frame-skip FRAME_SKIP]

    optional arguments:
      -h, --help            show this help message and exit
      --erode-amount ERODE_AMOUNT
                            Amount to erode face mask by
      --frame-skip FRAME_SKIP
                            Ratio of input frames to output frames

Options can alternatively be specified in a `pada.conf` in the working
directory, in the site config path, or global config path. To see the full list
of config paths run `pada.py print_config_paths`

## Requirements

`pada.py` requires `numpy`, `dlib`, `scipy`, `cv2`, and `appdirs`.

`make_vid.sh` requires `mencoder` and suitable codecs to be installed.

