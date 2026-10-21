import os
import numpy as np
import matplotlib.pyplot as plt

import utils_gen as utils

size = 100
group = 13
generator = utils.RandomLattice(supercell=[15,15,1], max_rotation_angle=0)
plot = utils.Plotting(path='./tests')


lattice, label = generator.generate_lattice(group)
crystal_render = generator.render_lattice(lattice)

plot.plot_lattice(crystal_render)

[rsx,rsy] = np.random.randint(-10, 10, size=2)
transform_r0, transform_i0 = generator.fourier_trasform_(255-crystal_render[500+rsx:1500+rsx,500+rsy:1500+rsy])
transform_r0, transform_i0 = transform_r0[350:650,350:650], transform_i0[350:650,350:650]

plot.plot_phaseF(transform_r0[50:250,50:250], filename='phaseF_r.png')
plot.plot_phaseF(transform_i0[50:250,50:250], filename='phaseF_i.png')

    