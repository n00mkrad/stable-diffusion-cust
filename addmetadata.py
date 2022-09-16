import os
import sys
import re
from PIL import PngImagePlugin
from PIL import Image
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--input",
    type=str,
    dest='inpath',
)
parser.add_argument(
    "-k",
    "--key",
    type=str,
    dest='key',
    default="Dream",
)
parser.add_argument(
    "-t",
    "--text",
    type=str,
    dest='text',
)

opt = parser.parse_args()

os.chdir(sys.path[0])

with Image.open(opt.inpath) as image:
    info = PngImagePlugin.PngInfo()
    info.add_text('Dream', opt.text)
    image.save(opt.inpath, 'PNG', pnginfo=info)

