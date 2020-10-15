"""
Find tags from:
    - a static image
    - a pre-recorded video
    - a web cam (opencv)
    - a picamera (on a pi)

Allow setting of detector parameters from command line
find_tags.py <source:fn, opencv:n, picamera>
    {-p parametername=value}  # detection parameters
    {-s setting=value}  # camera settings [only for picamera]
    {-h}  # run headless [no gui, wait for end of frames or KeyboardInterrupt]

Print out stats:
    - how many tags found (% of frames)
    - size distribution of tags
    - time to detect tags
"""

import argparse
import time

import cv2
import numpy

try:
    import picamera
    import picamera.array
    has_picamera = True
except ImportError:
    has_picamera = False


class Source:
    def __iter__(self):
        # yield frames or raise StopIterration
        raise StopIteration


class ImageSource(Source):
    def __init__(self, fn):
        self.fn = fn

    def __iter__(self):
        yield cv2.imread(self.fn, cv2.IMREAD_GRAYSCALE)
        #raise StopIteration


class VideoSource(Source):
    def __init__(self, video_id):
        self.cap = cv2.VideoCapture(video_id)

    def __iter__(self):
        while True:
            r, im = self.cap.read()
            if not r:
                raise StopIteration
            yield cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def convert_resolution(v):
    if isinstance(v, str):
        for sep in ',xX \t':
            tks = v.split(',')
            if len(tks) == 2:
                break
        if len(tks) != 2:
            raise ValueError("Unknown resolution format: %s" % v)
        w, h = int(tks[0]), int(tks[1])
    else:
        w, h = v
    if (w % 32) or (h % 16):
        w = (w // 32) * 32
        h = (h // 16) * 16
    return w, h


class PiCameraSource(Source):
    _special_settings = {
        'resolution': convert_resolution,
    }
    def __init__(self):
        if not has_picamera:
            raise RuntimeError("picamera is not available")
        self.cam = picamera.PiCamera()

    def configure(self, **settings):
        for k in settings:
            if k in self._special_settings:
                v = self._special_settings[k](settings[k])
            else:
                t = type(getattr(self.cam, k))
                v = t(settings[k])  # convert new setting to correct type
            setattr(self.cam, k, v)

    def __iter__(self):
        # make destination buffer
        w, h = self.cam.resolution
        im = numpy.empty((h, w, 3), dtype='uint8')
        while True:
            self.cam.capture(im, 'yuv', use_video_port=True)
            yield im[:, :, 0]


def make_source(source):
    if '.' in source:  # fn: image or video
        ext = source.split('.')[-1].lower()
        if ext in ('avi', 'mjpeg', 'mjpg', 'mpg', 'h264'):
            return VideoSource(source)
        else:
            return ImageSource(source)
    elif 'camera' in source:  # opencv cap
        if ':' in source:
            _, vid_id = source.split(':')
            vid_id = int(vid_id)
        else:
            vid_id = -1
        return VideoSource(vid_id)
    elif source == 'picamera':  # picamera
        if not has_picamera:
            raise RuntimeError("picamera is not available")
        return PiCameraSource()
    else:  # unknown source
        raise ValueError("Unknown source: %s" % source)


class TagFinder:
    def __init__(self, tag_dictionary=None, **kwargs):
        if tag_dictionary is None:
            tag_dictionary = '4X4_50'
        if isinstance(tag_dictionary, str):
            if 'DICT' not in tag_dictionary:
                tag_dictionary = "DICT_%s" % tag_dictionary
            tag_dictionary = tag_dictionary.upper()
            if not hasattr(cv2.aruco, tag_dictionary):
                raise ValueError("Unknown tag dictionary: %s" % tag_dictionary)
            tag_dictionary = getattr(cv2.aruco, tag_dictionary)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(tag_dictionary)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        if len(kwargs):
            print("Setting parameters...")
        for kw in kwargs:
            if not hasattr(self.aruco_params, kw):
                raise ValueError("Unknown parameter: %s" % kw)
            t = type(getattr(self.aruco_params, kw))  # get type to convert to
            v = t(kwargs[kw])
            setattr(self.aruco_params, kw, v)
            print("\t%s to %s" % (kw, v))
        self.reset_stats()

    def reset_stats(self):
        self.n_frames = 0
        self.dts = 0
        self.n_tags = 0
        self.areas_min = numpy.inf
        self.areas_max = -numpy.inf
        self.areas_sum = 0

    def process_grayscale_image(self, im):
        #im = a[:, :, 0]  # take y component
        t0 = time.monotonic()
        self.tags_results = cv2.aruco.detectMarkers(
            im, self.aruco_dict, parameters=self.aruco_params)
        t1 = time.monotonic()
        self.dts += t1 - t0
        nt = len(self.tags_results[0])
        if nt:
            self.n_tags += nt
            # compute tag areas
            areas = numpy.array([
                cv2.contourArea(tag_pts) for tag_pts in self.tags_results[0]])
            self.areas_sum += numpy.sum(areas)
            self.areas_min = min(self.areas_min, areas.min())
            self.areas_max = max(self.areas_max, areas.max())
        self.n_frames += 1

    def print_stats(self):
        print("Parameters:")
        for attr in sorted(dir(self.aruco_params)):
            if attr[0] == '_' or attr == 'create':
                continue
            print("\t%s=%s" % (attr, getattr(self.aruco_params, attr)))
        print("===")
        print("Processed %s frames" % (self.n_frames, ))
        if self.n_frames == 0:
            return
        print("Averages...")
        print("\t%0.4f seconds to detect tags" % (self.dts / self.n_frames, ))
        print("\t%0.2f average tags per frame" % (self.n_tags / self.n_frames, ))
        if self.n_tags:
            print("\t%0.2f minimum tag area" % (self.areas_min, ))
            print("\t%0.2f maximum tag area" % (self.areas_max, ))
            print("\t%0.2f average tag area" % (self.areas_sum / self.n_tags, ))


def draw_tags(im, tags_results):
    canvas = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    # candidate tags
    for pts in tags_results[2]:
        cv2.polylines(canvas, pts.astype('int32'), False, (0, 0, 255), 3)
    if not len(tags_results[0]):
        return canvas
    # mark tags
    cv2.aruco.drawDetectedMarkers(canvas, tags_results[0], tags_results[1])
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        help="Video/image source, can be: 'camera', 'picamera', or a filename")
    parser.add_argument(
        "-d", "--downsample", default=1, type=int,
        help="Downsample image by selecting every N-th pixel before display")
    parser.add_argument(
        "-H", "--headless", default=False, action="store_true",
        help="Run without GUI to show images and tag finding results")
    parser.add_argument(
        "-p", "--parameter", default=[], action="append",
        help="Detection parameters as key=value (ie detectInvertedMarker=True)")
    parser.add_argument(
        "-s", "--setting", default=[], action="append",
        help="picamera specific settings as key=value, where key is an attribute of the PiCamera class")
    parser.add_argument(
        "-t", "--tag_dictionary", default=None, type=str,
        help="Tag dictionary (as it appears in cv2.aruco)")

    args = parser.parse_args()

    # make image source
    source = make_source(args.source)
    if isinstance(source, PiCameraSource) and len(args.setting):
        settings = {}
        for s in args.setting:
            if "=" not in s:
                raise ValueError("Invalid setting missing '=': %s" % s)
            k, v = s.split("=")
            settings[k] = v
        source.configure(**settings)

    # make detector
    parameters = {}
    for p in args.parameter:
        if "=" not in p:
            raise ValueError("Invalid parameter missing '=': %s" % p)
        k, v = p.split("=")
        parameters[k] = v
    tf = TagFinder(args.tag_dictionary, **parameters)

    try:
        for im in source:
            tf.process_grayscale_image(im)
            if not args.headless:
                canvas = draw_tags(im, tf.tags_results)
                cv2.imshow('win', canvas[::args.downsample, ::args.downsample])
                k = cv2.waitKey(30)
                if k == ord('q'):
                    raise KeyboardInterrupt
                elif k == ord('r'):
                    print("Resetting stats")
                    tf.reset_stats()
        # no images left, sit on last frame after printing stats
        tf.print_stats()
        while True:
            if cv2.waitKey(30) == ord('q'):
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        tf.print_stats()



if __name__ == '__main__':
    main()
