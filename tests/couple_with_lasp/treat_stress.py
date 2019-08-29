#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np

box=np.loadtxt('./box.raw')
box=box.reshape(-1, 3)
volume=np.abs(np.dot(np.cross(box[0], box[1]),box[2]))
stress=np.loadtxt('./stress_tmp')
f_stress_all=open("stress_all.raw", "a")
np.savetxt(f_stress_all, stress.reshape(1, -1), fmt="%10.6f")
f_stress_all.close()
stress=stress/volume
np.savetxt('./stress_tmpp', stress, fmt="%10.6f")
