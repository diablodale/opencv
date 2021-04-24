#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class RemapFunctor(NewOpenCVTests):

    def test_oldfunction(self):
        src = self.get_sample("samples/data/chicky_512.png")
        self.assertIsInstance(src, np.ndarray)
        map_x = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
        map_y = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
        for i in range(map_x.shape[0]):
            map_x[i, :] = [x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]): # vertical flip
            map_y[:, j] = [map_y.shape[0]-y for y in range(map_y.shape[0])]

        dst = cv.remap_legacy(src, map_x, map_y, cv.INTER_LINEAR)
        self.assertEqual(src.shape[0], dst.shape[0])
        self.assertEqual(src.shape[1], dst.shape[1])

    def test_functor1(self):
        src = self.get_sample("samples/data/chicky_512.png")
        self.assertIsInstance(src, np.ndarray)
        map_x = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
        map_y = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
        for i in range(map_x.shape[0]):
            map_x[i, :] = [x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]): # vertical flip
            map_y[:, j] = [map_y.shape[0]-y for y in range(map_y.shape[0])]

        # create using unnamed params and use it -- all on one line
        dst = cv.remap(map_x, map_y, cv.INTER_LINEAR).run(src)

        self.assertEqual(src.shape[0], dst.shape[0])
        self.assertEqual(src.shape[1], dst.shape[1])

    def test_functor2(self):
        src = self.get_sample("samples/data/chicky_512.png")
        self.assertIsInstance(src, np.ndarray)
        map_x = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
        map_y = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
        for i in range(map_x.shape[0]):
            map_x[i, :] = [x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]): # vertical flip
            map_y[:, j] = [map_y.shape[0]-y for y in range(map_y.shape[0])]

        # create using named params and use it -- all on one line
        dst = cv.remap().map1(map_x).map2(map_y).interpolation(cv.INTER_LINEAR).run(src)

        self.assertEqual(src.shape[0], dst.shape[0])
        self.assertEqual(src.shape[1], dst.shape[1])

    def test_functor3(self):
        src = self.get_sample("samples/data/chicky_512.png")
        self.assertIsInstance(src, np.ndarray)
        map_x = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
        map_y = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
        for i in range(map_x.shape[0]):
            map_x[i, :] = [x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]): # vertical flip
            map_y[:, j] = [map_y.shape[0]-y for y in range(map_y.shape[0])]

        # create remapper once and reuse
        remapper = cv.remap().map1(map_x).map2(map_y).interpolation(cv.INTER_LINEAR)
        for idx in range(100):
            dst = remapper.run(src)
            self.assertEqual(src.shape[0], dst.shape[0])
            self.assertEqual(src.shape[1], dst.shape[1])

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
