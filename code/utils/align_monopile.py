#!/usr/bin/env python3
"""
align_monopile.py

Align a seabed point cloud around a monopile foundation:
1. Fit a plane to the seabed using RANSAC → rotate so seabed is horizontal
2. Detect the circular hole left by the removed monopile → center it at origin

Output: denoised_aligned.ply — rotated & translated point cloud with:
  - Monopile hole center at (0, 0, 0)
  - Seabed plane horizontal (normal along Z axis)
"""

import numpy as np
from scipy.spatial import cKDTree

# ── Configurable parameters ─────────────────────────────────────────────────

RANSAC_THRESHOLD  = 0.20   # [m] inlier distance for plane fit
RANSAC_ITERATIONS = 2000
MIN_HOLE_RADIUS   = 0.05   # [m] minimum expected hole radius
MAX_HOLE_RADIUS   = 2.00   # [m] maximum expected hole radius


# ── PLY I/O ─────────────────────────────────────────────────────────────────

def load_ply(filepath):
    """Load an ASCII PLY file.

    Returns
    -------
    points : (N, 3) float64   xyz coordinates
    normals : (N, 3) float64  normal vectors (may be all-zero)
    colors : (N, 3) uint8     rgb values
    alpha : (N,) uint8        alpha channel
    header_lines : list[str]  raw header block (for rewriting)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse header
    header_lines = []
    num_vertices = 0
    properties = []
    in_header = True

    for i, line in enumerate(lines):
        if in_header:
            header_lines.append(line)
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                properties.append((parts[2], parts[1]))
            elif line.startswith('end_header'):
                in_header = False
                header_end = i + 1
                break

    # Read data
    data = np.loadtxt(lines[header_end:header_end + num_vertices], dtype=np.float64)

    col = 0
    points = data[:, col:col+3]; col += 3
    normals = data[:, col:col+3]; col += 3
    colors = data[:, col:col+3].astype(np.uint8); col += 3
    alpha = data[:, col].astype(np.uint8) if col < data.shape[1] else None

    print(f"  Loaded {num_vertices} points from {filepath}")
    return points, normals, colors, alpha, header_lines


def save_ply(filepath, points, normals, colors, alpha, header_lines):
    """Save a transformed point cloud as ASCII PLY, reusing the original header."""
    # Build output columns: x y z nx ny nz r g b [alpha]
    cols = [points, normals, colors]
    if alpha is not None:
        cols.append(alpha.reshape(-1, 1))
    output = np.column_stack(cols)

    # Build format string matching the data columns
    # points: 3 floats, normals: 3 floats, colors: 3 ints, alpha: 1 int = 10 columns
    fmt_parts = ['%.6f'] * 3 + ['%.6f'] * 3 + ['%d'] * 3
    if alpha is not None:
        fmt_parts.append('%d')
    fmt = ' '.join(fmt_parts)

    with open(filepath, 'w') as f:
        for line in header_lines:
            f.write(line)
        np.savetxt(f, output, fmt=fmt)

    print(f"  Saved {len(points)} points to {filepath}")


# ── Plane fitting ───────────────────────────────────────────────────────────

def fit_plane_ransac(points, threshold=RANSAC_THRESHOLD,
                     max_iters=RANSAC_ITERATIONS):
    """Fit a plane to *points* using RANSAC, then refine with SVD.

    Returns
    -------
    normal : (3,) float64   unit plane normal
    d : float               plane offset: <normal, p> + d = 0
    inlier_mask : (N,) bool
    """
    N = len(points)
    best_inliers = np.array([], dtype=int)
    best_normal = None
    best_d = 0.0

    rng = np.random.default_rng(42)
    all_triplets = rng.integers(0, N, size=(max_iters, 3))

    for triplet in all_triplets:
        p0, p1, p2 = points[triplet]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        nrm = np.linalg.norm(normal)
        if nrm < 1e-12:
            continue
        normal /= nrm

        d = -np.dot(normal, p0)
        dists = np.abs(points @ normal + d)
        inliers = np.flatnonzero(dists < threshold)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = normal.copy()
            best_d = d

    # Refit using all inliers via SVD (more accurate)
    if len(best_inliers) > 3:
        inlier_pts = points[best_inliers]
        centroid = inlier_pts.mean(axis=0)
        _, _, vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        best_normal = vt[2]  # smallest singular vector = plane normal
        best_normal /= np.linalg.norm(best_normal)
        best_d = -np.dot(best_normal, centroid)

    inlier_mask = np.zeros(N, dtype=bool)
    inlier_mask[best_inliers] = True

    return best_normal, best_d, inlier_mask


# ── Rotation ────────────────────────────────────────────────────────────────

def rotation_align_vector_to_z(v):
    """Compute a 3×3 rotation matrix R such that R @ v is parallel to +Z.

    Uses Rodrigues' rotation formula:
        R = I + K + K²/(1+cos(θ))

    where K is the cross-product matrix of  (v × z)  — the **unnormalized**
    rotation axis.  (|v × z| = sin(θ), so K = sin(θ)·K_hat.)
    """
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])

    # Unnormalized axis:  a = v × z   (magnitude = sin θ)
    axis = np.cross(v, z)
    sin_theta = np.linalg.norm(axis)
    cos_theta = np.dot(v, z)

    if sin_theta < 1e-12:
        # v already parallel to +Z (cos ≈ 1) → identity
        return np.eye(3)

    if np.abs(cos_theta + 1.0) < 1e-12:
        # v is antiparallel to Z → rotate 180° around any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(perp, v)) > 0.99:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(v, perp)
        axis /= np.linalg.norm(axis)
        K = cross_product_matrix(axis)
        R = np.eye(3) + 2 * K @ K
        return R

    # K = cross-product matrix of UNNORMALIZED axis (K = sinθ · K̂)
    K = cross_product_matrix(axis)
    R = np.eye(3) + K + (K @ K) / (1.0 + cos_theta)
    return R


def cross_product_matrix(a):
    """Cross-product matrix K such that K @ x = a × x."""
    return np.array([[0.0, -a[2],  a[1]],
                     [a[2],  0.0, -a[0]],
                     [-a[1], a[0],  0.0]])


# ── Hole detection ──────────────────────────────────────────────────────────

def find_hole_voronoi_local(xy_points, search_radius=0.8,
                             min_radius=MIN_HOLE_RADIUS,
                             max_radius=MAX_HOLE_RADIUS):
    """Find the largest empty circle near the XY mean using Voronoi vertices.

    The largest empty circle in a point set has its centre at a Voronoi
    vertex.  We constrain the search to vertices within *search_radius* of
    the initial XY mean to target the monopile hole (not gaps elsewhere).

    Returns
    -------
    center_xy : (2,) float64 | None
    radius : float | None
    """
    from scipy.spatial import Voronoi

    pts = xy_points
    init_center = pts.mean(axis=0)
    print(f"  Initial centre (XY mean): ({init_center[0]:.4f}, {init_center[1]:.4f})")
    print(f"  Voronoi search radius:    {search_radius:.3f} m")

    # Voronoi diagram — O(N log N), fine for 36k points
    vor = Voronoi(pts)

    best_radius = 0.0
    best_vertex = None

    n_checked = 0
    for vertex in vor.vertices:
        d2c = np.sqrt(((vertex - init_center) ** 2).sum())
        if d2c > search_radius:
            continue
        n_checked += 1

        # Distance to nearest data point = radius of empty circle
        # Use cKDTree for speed on all Voronoi vertices?  Actually, each
        # Voronoi vertex is equidistant to ≥ 3 points by definition.
        # The radius is the distance to any of its generating sites.
        # But we don't easily have the vertex→site mapping from scipy.
        # Compute directly (fast enough for the local subset).
        dist = np.sqrt(((pts - vertex) ** 2).sum(axis=1)).min()

        if dist > best_radius:
            best_radius = dist
            best_vertex = vertex.copy()

    print(f"  Voronoi vertices checked: {n_checked}")
    if best_vertex is None:
        print("  No Voronoi vertex found in search radius")
        return None, None

    print(f"  Largest empty circle — centre: ({best_vertex[0]:.4f}, "
          f"{best_vertex[1]:.4f}), radius: {best_radius:.4f} m")

    # ── Quality check ────────────────────────────────────────────────
    dx2 = pts[:, 0] - best_vertex[0]
    dy2 = pts[:, 1] - best_vertex[1]
    dists2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
    n_in_hole = (dists2 < best_radius * 0.8).sum()
    print(f"  Points inside hole (r < {0.8 * best_radius:.3f} m): {n_in_hole}")

    if best_radius < min_radius or best_radius > max_radius:
        print(f"  WARNING: Radius {best_radius:.4f} outside expected range "
              f"[{min_radius}, {max_radius}]")

    return best_vertex, best_radius


# ── Circle fitting (for refinement / fallback) ─────────────────────────────

def fit_circle_algebraic(pts_2d):
    """Algebraic least-squares circle fit to boundary points.

    Solves  ‖x²+y² + a·x + b·y + c‖ → min.
    Then  center = (-a/2, -b/2),  radius = √(cx² + cy² − c).
    """
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    rhs = -(x ** 2 + y ** 2)
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    a, b_val, c = sol
    cx = -a / 2.0
    cy = -b_val / 2.0
    radius = np.sqrt(max(cx * cx + cy * cy - c, 0.0))
    return np.array([cx, cy]), radius


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    input_file  = '/home/wyh/Desktop/workflow/random_145_h6_1_4/points3D.ply'
    output_file = '/home/wyh/Desktop/workflow/random_145_h6_1_4/points3D.ply'

    # ── 1. Load ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1 — Loading point cloud")
    points, normals, colors, alpha, header = load_ply(input_file)

    # ── 2. Fit seabed plane ─────────────────────────────────────────────
    print("\nStep 2 — Fitting seabed plane (RANSAC)")
    normal, d, inlier_mask = fit_plane_ransac(points)
    print(f"  Plane normal:  ({normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f})")
    print(f"  Plane offset:  d = {d:.4f}")
    print(f"  Inliers:       {inlier_mask.sum()} / {len(points)} "
          f"({100 * inlier_mask.mean():.1f} %)")
    tilt_deg = np.arccos(np.abs(np.dot(normal, [0, 0, 1]))) * 180 / np.pi
    print(f"  Seabed tilt:   {tilt_deg:.1f}° from horizontal")

    # ── 3. Rotate to horizontal ────────────────────────────────────────
    print("\nStep 3 — Rotating seabed to horizontal")
    R = rotation_align_vector_to_z(normal)
    # Verify
    new_normal = R @ normal
    print(f"  Rotation check — R·n = ({new_normal[0]:.8f}, {new_normal[1]:.8f}, "
          f"{new_normal[2]:.8f})  (should be ≈ [0, 0, 1])")

    rotated_pts = points @ R.T            # (N, 3)
    rotated_normals = normals @ R.T       # (N, 3)

    # ── 4. Detect hole ─────────────────────────────────────────────────
    print("\nStep 4 — Detecting monopile hole (local Voronoi)")
    xy = rotated_pts[:, :2]
    center_xy, radius = find_hole_voronoi_local(xy)

    if center_xy is None:
        print("  Hole detection failed, falling back to XY mean")
        center_xy = xy.mean(axis=0)
        radius = 0.0

    # ── 5. Estimate hole-center Z via local plane fit ──────────────────
    print("\nStep 5 — Estimating Z at hole center (local plane fit)")
    # Collect points in an annulus just outside the hole
    dx = rotated_pts[:, 0] - center_xy[0]
    dy = rotated_pts[:, 1] - center_xy[1]
    dists_xy = np.sqrt(dx * dx + dy * dy)

    if radius > 0.0:
        annulus = (dists_xy >= radius * 0.9) & (dists_xy <= radius * 2.0)
        annulus_pts = rotated_pts[annulus]
        if len(annulus_pts) > 10:
            # Fit a local plane to the annulus points: Z = a*X + b*Y + c
            A = np.column_stack([annulus_pts[:, 0], annulus_pts[:, 1],
                                 np.ones(len(annulus_pts))])
            z_vals = annulus_pts[:, 2]
            coeffs, *_ = np.linalg.lstsq(A, z_vals, rcond=None)
            a, b, c = coeffs
            center_z = a * center_xy[0] + b * center_xy[1] + c
            print(f"  Local plane: Z = {a:.4f}·X + {b:.4f}·Y + {c:.4f}")
            print(f"  Annulus points used: {len(annulus_pts)}")
        else:
            # Fallback: median Z of nearest points
            tree_xy = cKDTree(xy)
            _, idx = tree_xy.query(center_xy, k=50)
            center_z = float(np.median(rotated_pts[idx, 2]))
    else:
        tree_xy = cKDTree(xy)
        _, idx = tree_xy.query(center_xy, k=50)
        center_z = float(np.median(rotated_pts[idx, 2]))

    print(f"  Hole center Z:  {center_z:.4f}")
    center_3d = np.array([center_xy[0], center_xy[1], center_z])
    print(f"  Hole center 3D: ({center_3d[0]:.4f}, {center_3d[1]:.4f}, "
          f"{center_3d[2]:.4f})")

    # ── 6. Translate ───────────────────────────────────────────────────
    print("\nStep 6 — Translating hole center to origin")
    aligned_pts = rotated_pts - center_3d

    # ── 7. Verify ──────────────────────────────────────────────────────
    print("\nStep 7 — Verification")
    xy_a = aligned_pts[:, :2]
    dists_from_origin = np.sqrt((xy_a ** 2).sum(axis=1))
    min_dist = dists_from_origin.min()
    print(f"  Post-transform XY mean:           ({xy_a[:, 0].mean():.6f}, "
          f"{xy_a[:, 1].mean():.6f})")
    print(f"  Min distance from origin (XY):    {min_dist:.4f} m "
          f"(≈ hole radius)")
    print(f"  Mean distance from origin (XY):   {dists_from_origin.mean():.4f} m")
    print(f"  Z range:                          [{aligned_pts[:, 2].min():.4f}, "
          f"{aligned_pts[:, 2].max():.4f}]")
    print(f"  Z mean:                           {aligned_pts[:, 2].mean():.6f}")
    print(f"  Z std:                            {aligned_pts[:, 2].std():.6f}")

    # ── 8. Save ────────────────────────────────────────────────────────
    print("\nStep 8 — Saving")
    save_ply(output_file, aligned_pts, rotated_normals, colors, alpha, header)
    print("\nDone.")


if __name__ == '__main__':
    main()
