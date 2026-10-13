# -*- coding: utf-8 -*-
# Port of TopoMultiFile1xiuzheng.m (Dr. Lv) to Python.
# Applies the per-file correction chain to each .ply (x,y,z only),
# then the global pile shift.  Writes 6 corrected .ply files.
import glob, os, math
import numpy as np

# ---- parameters from the MATLAB code ----
A     = [-1, 2]           # x range / m  (post axis-swap)
B     = [-1, 2]           # y range / m
C     = [-1, 0.3]         # z range / m (height filter)
deltax = [0, -0.49, -0.543, -0.277, 2.416, 2.424]   # splice calibration / m
deltay = [0, -0.349, -0.033, -0.4265, 0.111, 1.375]
deltaz = [0, 0.172, 0.009, 0.184, 0.004, 0.007]
Angle  = [217, 290]       # heading angle range / deg
H      = 0.5              # sensor head height / m
Gama   = 1                # head tilt calibration / deg
Pile   = [0.81, 0.80]     # pile position / m (subtracted at the end)

def rot2d(deg):
    """CCW 2x2 rotation matrix, same as cosd/sind version in MATLAB."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s], [s, c]])

def read_ply_xyz(fn):
    with open(fn) as f:
        lines = f.readlines()
    # skip the ascii header up to end_header
    idx = next(i for i, l in enumerate(lines) if l.strip() == 'end_header')
    return np.loadtxt(lines[idx + 1:])            # N x 3

def write_ply_xyz(fn, pts):
    pts = np.atleast_2d(pts)
    with open(fn, 'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\n' % pts.shape[0])
        f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(f, pts, fmt='%.10f')

def correct(xyz, i):
    """xyz : N x 3 array holding the ORIGINAL x,y,z columns (Data(:,1:3))."""
    x = xyz[:, 0].astype(float).copy()
    y = xyz[:, 1].astype(float).copy()
    z = xyz[:, 2].astype(float).copy()

    # ---- axis swap / rotation of raw scanner frame ----
    nx = -z
    ny = -y
    nz = -x

    # ---- crop out of x/y range ----
    mask = (nx >= A[0]) & (nx <= A[1]) & (ny >= B[0]) & (ny <= B[1])
    nx, ny, nz = nx[mask], ny[mask], nz[mask]

    # ---- shift origin to min ----
    nx = nx - nx.min()
    ny = ny - ny.min()

    # ---- heading-angle correction, rotation in the y-z plane ----
    theta = -(Angle[1] - Angle[0]) / 5.0           # -14.6 deg
    R = rot2d(theta)
    yz = R @ np.vstack([ny, nz])
    ny, nz = yz[0], yz[1]
    nz = nz + H                                     # add head height

    # ---- tilt (Gama) correction, rotation in the x-z plane ----
    R = rot2d(Gama)
    xz = R @ np.vstack([nx, nz])
    nx, nz = xz[0], xz[1]

    # ---- height filter ----
    mask = (nz >= C[0]) & (nz <= C[1])
    nx, ny, nz = nx[mask], ny[mask], nz[mask]

    # ---- origin calibration + per-file splice delta ----
    nx = nx + nx.max() + deltax[i]
    ny = ny + ny.max() + deltay[i]
    nz = nz + deltaz[i]

    # ---- global pile shift (in MATLAB applied to the merged cloud) ----
    nx = nx - Pile[0]
    ny = ny - Pile[1]

    return np.column_stack([nx, ny, nz])

# process the 6 files in MATLAB dir() alphabetical order == sorted glob
stems = [os.path.splitext(os.path.basename(f))[0]
         for f in sorted(glob.glob('*.xyz'))]

for i, stem in enumerate(stems):
    src = stem + '.ply'
    dst = stem + '_corrected.ply'
    raw = read_ply_xyz(src)
    out = correct(raw, i)
    write_ply_xyz(dst, out)
    print('%-8s  in=%6d  ->  out=%6d   %s' %
          (stem, raw.shape[0], out.shape[0], os.path.basename(dst)))
