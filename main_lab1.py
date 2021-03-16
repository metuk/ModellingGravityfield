#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:19:06 2021

@author: david
"""

import numpy as np
n = np.load("normals_2008-05.rightHandSide.npy")
N = np.load("normals_2008-05.npy")
x_hat = np.linalg.inv(N)@n