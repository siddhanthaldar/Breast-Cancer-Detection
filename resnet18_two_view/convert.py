# Convert from lpeg to jpg
import os
import sys
import re
import subprocess
import numpy
import logging

ROOT_DIR = os.path.abspath("../")
# print(ROOT_DIR)
BIN = os.path.join(ROOT_DIR,"ljpeg-ddsm", "jpegdir", "jpeg")
# print(BIN)

if not os.path.exists(BIN):
    print("jpeg is not built yet; use 'cd jpegdir; make' first")
    sys.exit(0)

PATTERN = re.compile('\sC:(\d+)\s+N:(\S+)\s+W:(\d+)\s+H:(\d+)\s')
# print("pattern : ",PATTERN)

def read (path):
    cmd = '%s -d -s %s' % (BIN, path)
    l = subprocess.check_output(cmd, shell=True)
    #print l
    m = re.search(PATTERN, l)
    C = int(m.group(1)) # I suppose this is # channels
    F = m.group(2)
    W = int(m.group(3))
    H = int(m.group(4))
    assert C == 1
    im = numpy.fromfile(F, dtype='uint16').reshape(H, W)
    L = im >> 8
    H = im & 0xFF
    im = (H << 8) | L
    os.remove(F)
    return im    

import glob
import cv2

def ljpeg_to_jpg(ljpeg, output, scale, verify=True, visual=True):
    path = ljpeg
    tiff = output

    assert 'LJPEG' in path

    root = os.path.dirname(path)
    stem = os.path.splitext(path)[0]

    # read ICS
    ics = glob.glob(root + '/*.ics')[0]
    name = path.split('.')[-2]

    W = None
    H = None
    # find the shape of image
    for l in open(ics, 'r'):
        l = l.strip().split(' ')
        if len(l) < 7:
            continue
        if l[0] == name:
            W = int(l[4])
            H = int(l[2])
            bps = int(l[6])
            if bps != 12:
                logging.warn('BPS != 12: %s' % path)
            break

    # assert W != None
    # assert H != None

    # count = 1
    if W == None or H == None :
    #     count = 0
    	print("\n *********** Entered ************ \n")
    #     print(count)
    	return None

    # count = 2
    # # print("\n *********** came out ************ \n")
    # print(count)
    image = read(path)

    if W != image.shape[1]:
        logging.warn('reshape: %s' % path)
        image = image.reshape((H, W))

    raw = image

    if visual:
        logging.warn("normalizing color, will lose information")
        if verify:
            logging.error("verification is going to fail")
        if scale:
            rows, cols = image.shape
            image = cv2.resize(image, (int(cols * scale), int(rows * scale)))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = numpy.uint8(image)
    elif scale:
        logging.error("--scale must be used with --visual")
        sys.exit(1)
        #image = cv2.equalizeHist(image)
    #tiff = stem + '.TIFF'
    # cv2.imwrite(tiff, image)
    # cv2.imshow('tiff', image)
    # cv2.waitKey(0)

    if verify:
        verify = cv2.imread(tiff, -1)
        if numpy.all(raw == verify):
            logging.info('Verification successful, conversion is lossless')
        else:
            logging.error('Verification failed: %s' % path)
            
    return image
