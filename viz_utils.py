import matplotlib
matplotlib.use('Agg')

import numpy as np
import pylab as pl
import nibabel as nb

# Utilities for colormaps
from matplotlib import cm as _cm
from matplotlib import colors as _colors

from scipy.stats import scoreatpercentile
from sklearn.preprocessing import StandardScaler
from nilearn.masking import _smooth_array
from nipy.labs.viz import plot_map
from nipy.labs.viz_tools import cm
from nipy.labs.viz_tools import anat_cache


def alpha_cmap(color):
    """ Return a colormap with the given color, and alpha going from
        zero to 1.
    """
    red, green, blue = color[:3]
    cmapspec = [(red, green, blue, 0.),
                (red, green, blue, 1.),
               ]
    cmap = _colors.LinearSegmentedColormap.from_list(
                                'alpha', cmapspec, _cm.LUTSIZE)
    cmap._init()
    cmap._lut[:, -1] = np.linspace(.5, .75, cmap._lut.shape[0])
    cmap._lut[-1, -1] = 0
    return cmap


def plot_bg(cut_coords=None, title=None):
    anat, anat_affine, anat_max = anat_cache._AnatCache.get_anat()
    figure = pl.figure(figsize=(8, 2.6), facecolor='w', edgecolor='w')
    ax = pl.axes([.0, .0, .85, 1], axisbg='w')
    slicer = plot_map(anat,
                      anat_affine,
                      cmap=pl.cm.gray,
                      vmin=.1 * anat_max,
                      vmax=.8 * anat_max,
                      figure=figure,
                      cut_coords=cut_coords,
                      axes=ax, )
    slicer.annotate()
    slicer.draw_cross()
    if title:
        slicer.title(title, x=.05, y=.9)
    return slicer


def plot_contour_atlas(niimgs, labels, cut_coords=None,
                     title=None, percentile=99):
    legend_lines = []
    slicer = plot_bg(cut_coords, title)
    atlas = np.vstack([niimg.get_data()[np.newaxis] for niimg in niimgs])
    # atlas = StandardScaler().fit_transform(atlas.T).T
    affine = niimgs[0].get_affine()
    for i, (label, data) in enumerate(zip(labels, atlas)):
        data = np.array(_smooth_array(data, affine, 5), copy=True)
        data[data < 0] = 0
        color = np.array(pl.cm.Set1(float(i) / (len(labels) - 1)))
        # data, affine = niimg.get_data(), niimg.get_affine()
        # affine = niimg.get_affine()
        level = scoreatpercentile(data.ravel(), percentile)
        slicer.contour_map(data, affine, levels=(level, ),
                           linewidth=2.5, colors=(color, ))
        slicer.plot_map(data, affine, threshold=level,
                        cmap=alpha_cmap(color))
        legend_lines.append(pl.Line2D([0, 0], [0, 0],
                            color=color, linewidth=4))

    ax = slicer.axes['z'].ax.get_figure().add_axes([.80, .1, .15, .8])
    pl.axis('off')
    ax.legend(legend_lines, labels, loc='center right',
              prop=dict(size=4), title='Labels',
              borderaxespad=0,
              bbox_to_anchor=(1 / .85, .5))
    return nb.Nifti1Image(atlas, affine=affine)


def plot_label_atlas(niimgs, labels, cut_coords=None, title=None):
    slicer = plot_bg(cut_coords, title)
    n_maps = len(niimgs)

    data = np.array([niimg.get_data() for niimg in niimgs])
    affine = niimgs[0].get_affine()
    mask = np.any(data, axis=0)
    atlas = np.ones(mask.shape, dtype='int') * -1
    atlas[mask] = np.argmax(np.abs(data), axis=0)[mask]
    colors = (np.arange(n_maps) + 1) / float(n_maps)
    colors = np.hstack([colors, [0]])
    slicer.plot_map(np.ma.masked_equal(colors[atlas], 0),
                    affine,
                    cmap=pl.cm.spectral, )

    legend_lines = [pl.Line2D([0, 0], [0, 0],
                    color=pl.cm.spectral(color), linewidth=4)
                    for color in colors]

    ax = slicer.axes['z'].ax.get_figure().add_axes([.80, .1, .15, .8])
    pl.axis('off')
    ax.legend(legend_lines, labels, loc='center right',
              prop=dict(size=4), title='Labels',
              borderaxespad=0,
              bbox_to_anchor=(1 / .85, .5))
    return nb.Nifti1Image(atlas, affine=affine)


if __name__ == '__main__':
    import os
    import glob
    import nibabel as nb

    data_dir = '/tmp/reporter'
    niimgs = [nb.load(img)
              for img in glob.glob(os.path.join(data_dir, '*.nii.gz'))]
    labels = [os.path.split(img)[1].split('.nii.gz')[0]
              for img in glob.glob(os.path.join(data_dir, '*.nii.gz'))]
    # plot_contour_atlas(niimgs, labels, (61, -20, 3), 'atlas', 99)
    plot_label_atlas(niimgs, labels, (61, -20, 3), 'atlas')

    pl.show()
