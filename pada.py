#!/usr/bin/env python

# Copyright (c) 2016 Matthew Earl
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


import argparse
import glob
import json
import logging
import os
import sys

import pada.align
import pada.framedrop
import pada.landmarks
import pada.logging


APP_NAME = "pada"
APP_AUTHOR = "matthewearl"
CONFIG_FILE_NAME = "pada.conf"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', help='Print debug information',
                        action='store_true')
    parser.add_argument('--config', help='Config file path', type=unicode,
                        default=CONFIG_FILE_NAME)
    parser.add_argument('--aligned-path',
                        help='Path where aligned images will be stored')
    parser.add_argument('--predictor-path',
                        help='DLib face predictor dat file',
                        type=unicode)
    parser.add_argument('--filtered-files',
                        help='File to write filtered files to',
                        type=unicode)

    subparsers = parser.add_subparsers(help='Sub-command help')

    print_config_paths = subparsers.add_parser(
        'print_config_paths',
        help='print config paths and exit')
    print_config_paths.set_defaults(cmd='print_config_paths')

    align_parser = subparsers.add_parser('align',
                                         help='align a set of images')
    align_parser.add_argument('--input-glob',
                              help='Input files glob', type=unicode)
    align_parser.add_argument('--img-thresh',
                              help='Max duplicate frame delta', type=float)
    align_parser.set_defaults(cmd='align')

    framedrop_parser = subparsers.add_parser(
                                       'framedrop',
                                       help='Drop frames from a set of images')
    framedrop_parser.add_argument('--erode-amount',
                              help='Amount to erode face mask by', type=int)
    framedrop_parser.add_argument(
                            '--frame-skip',
                            help='Ratio of input frames to output frames',
                            type=int)
    framedrop_parser.set_defaults(cmd='framedrop')

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()

    if cli_args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Build up a list of potential config files to parse.
    config_paths = []
    try:
        import appdirs
    except ImportError:
        logging.warn("appdirs not installed, not reading site/user config")
    else:
        config_paths.append(
            os.path.join(
                appdirs.user_config_dir(APP_NAME, APP_AUTHOR),
                CONFIG_FILE_NAME))
        config_paths.append(
            os.path.join(
                appdirs.site_config_dir(APP_NAME, APP_AUTHOR),
                CONFIG_FILE_NAME))
    if cli_args.config:
        config_paths.append(cli_args.config)

    if cli_args.cmd == "print_config_paths":
        for config_path in config_paths:
            print config_path
        sys.exit(0)

    # Attempt to open each config file, and update the `cfg` dict with each
    # one.
    cfg = {}
    for config_path in config_paths:
        try:
            with open(config_path) as f:
                d = json.load(f)
                if 'global' in d:
                    cfg.update(d['global'])
                if cli_args.cmd in d:
                    cfg.update(d[cli_args.cmd])
        except IOError:
            logging.warn("Could not open config file %s", config_path)
        else:
            logging.info("Read config file %s", config_path)

    # Add CLI arguments to the `cfg` dict.
    cfg.update((k, v) for k, v in cli_args.__dict__.items() if v is not None)
    logging.debug("Config is %r", cfg)

    # Execute the command by deferring to the appopriate module.
    landmark_finder = pada.landmarks.LandmarkFinder(
                                     os.path.expanduser(cfg['predictor_path']))
    if cli_args.cmd == "align":
        pada.align.align_images(
            input_files=sorted(glob.glob(cfg['input_glob'])),
            out_path=cfg['aligned_path'],
            landmark_finder=landmark_finder,
            img_thresh=cfg['img_thresh'])
    elif cli_args.cmd == "framedrop":
        filtered_files = pada.framedrop.filter_files(
            input_files=sorted(glob.glob(
                                  os.path.join(cfg['aligned_path'], '*.jpg'))),
            frame_skip=cfg['frame_skip'],
            erode_amount=cfg['erode_amount'],
            landmark_finder=landmark_finder)

        with open(cfg['filtered_files'], 'w') as f:
            for fname in filtered_files:
                f.write("{}\n".format(fname))

