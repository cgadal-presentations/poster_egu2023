import os
import sys
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import PyThemes.Beamer_169 as Beamer

# plt.switch_backend('agg')
plt.rcParams['figure.constrained_layout.h_pad'] = 0
plt.rcParams['figure.constrained_layout.w_pad'] = 0

# Paths to adjust before starting scripts
# where experimental directories are stored
path_manip = "/media/cyril/LaCie_orang/grand_canal/spring_summer2022/data/tests/18052022/manip"

#
list_image = sorted(glob.glob(os.path.join(path_manip, '*.tif')))
image_ref = np.array([np.array(Image.open(img))
                     for img in list_image[:10]]).mean(axis=0)
aspect = image_ref.shape[0]/image_ref.shape[1]

image_path = list_image[1200]
# parameters
figwidth = 0.45*Beamer.fig_width
fig, ax = plt.subplots(1, 1, constrained_layout=True,
                       figsize=(figwidth, aspect*figwidth))

image = np.array(Image.open(image_path))
image = -np.log(image/image_ref)
ax.set_xticks([])
ax.set_yticks([])
plt.imshow(image[:, ::-1], cmap='gray', vmin=0, vmax=4)
# plt.colorbar(location='top', label=r'$\textup{log}(I/I_{0})$')
plt.savefig('../img_steady_input.svg', dpi=1000)
