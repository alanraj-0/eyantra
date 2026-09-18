#!/usr/bin/env python3
"""
Task 1A - Khoj-o-Drone
Detects ArUco markers, rectifies the arena to a fixed top-down canvas,
locates survivors by color, and writes the results file.

Usage:
    python3 task1a.py --image path/to/image.jpg
"""

import argparse
import sys

import cv2
import numpy as np

MARKER_IDS = {80, 85, 90, 95}
CANVAS_SIZE = 900
GRID_CELLS = 12          # 12x12 cells -> 11x11 = 121 interior intersections
LINES = GRID_CELLS - 1   # number of interior grid lines in each direction


def load_image(path):
    """Just a wrapper around cv2.imread with a loud failure instead of a
    silent None that would crash three lines later with a confusing error."""
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: could not load image at '{path}'", file=sys.stderr)
        sys.exit(1)
    return img


def detect_markers(img):
    """
    ArUco markers are square binary codes from a known dictionary
    (here: 4x4 bits, 250 possible IDs). Detection = find quadrilateral
    contours in the image, warp each candidate to a flat square, and
    check whether its bit pattern matches an entry in the dictionary.

    CORNER_REFINE_SUBPIX matters mathematically: raw contour corners are
    only accurate to the nearest whole pixel. Subpixel refinement looks
    at the local image gradient around each corner and finds the point
    where the gradient direction is most consistent with a "true" corner
    - effectively fitting to a fraction of a pixel instead of rounding.
    Since 4 corners here become the input to a homography (next step),
    a half-pixel error at this stage gets amplified across the whole
    900x900 rectified image, so this refinement is not optional.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    found = set(ids.flatten().tolist()) if ids is not None else set()
    if not MARKER_IDS.issubset(found):
        print(f"ERROR: expected markers {sorted(MARKER_IDS)}, found {sorted(found)}",
              file=sys.stderr)
        sys.exit(1)

    # corners[i] has shape (1, 4, 2): 4 (x, y) corner points for ids[i],
    # in order [top-left, top-right, bottom-right, bottom-left] *relative
    # to that marker's own printed orientation* - not relative to the arena.
    return {int(mid): c[0] for c, mid in zip(corners, ids.flatten())
            if int(mid) in MARKER_IDS}


def order_corners(markers):
    """
    Figures out which physical corner of the ARENA each marker sits at
    (top-left / top-right / bottom-right / bottom-left), using geometry
    only - never the marker's ID number. This is what makes the pipeline
    work on any image, regardless of which ID happens to be placed where
    or how the camera is rotated.

    The math (a classic "order 4 points" trick):
    For each marker center (x, y), compute two derived values:
        s = x + y      (sum)
        d = x - y      (difference)
    Why this works: imagine the 4 points roughly forming a rectangle.
      - The point with the SMALLEST sum (x+y) is the one closest to the
        origin in both x and y at once -> top-left.
      - The point with the LARGEST sum is farthest in both x and y at
        once -> bottom-right.
      - The point with the LARGEST difference (x-y) has a large x but a
        small y -> top-right.
      - The point with the SMALLEST difference has a small x but a large
        y -> bottom-left.
    This holds for any rotation/skew where the rectangle hasn't been
    flipped inside-out, so it's robust to the camera angle changing
    between images - unlike hardcoding "marker 80 = top-left".
    """
    centers = {mid: pts.mean(axis=0) for mid, pts in markers.items()}
    tl_id = min(centers, key=lambda k: centers[k][0] + centers[k][1])  # min(x+y)
    br_id = max(centers, key=lambda k: centers[k][0] + centers[k][1])  # max(x+y)
    tr_id = max(centers, key=lambda k: centers[k][0] - centers[k][1])  # max(x-y)
    bl_id = min(centers, key=lambda k: centers[k][0] - centers[k][1])  # min(x-y)

    # Use each marker's own boundary-facing corner (not its center) as the
    # actual arena-corner point, so the rectified canvas captures the full
    # arena instead of cropping inward by half a marker's width.
    src = np.array([
        markers[tl_id][0],  # that marker's own top-left corner
        markers[tr_id][1],  # top-right corner
        markers[br_id][2],  # bottom-right corner
        markers[bl_id][3],  # bottom-left corner
    ], dtype=np.float32)
    return src


def rectify(img, src_pts):
    """
    Perspective (homography) transform: maps the 4 arena-corner points in
    the original, tilted photo to the 4 corners of a flat 900x900 square.

    Math: a homography H is a 3x3 matrix (8 degrees of freedom - the 9th
    entry is fixed by scale) that maps a point via homogeneous coords:
        [x' y' w']^T = H @ [x y 1]^T
        final pixel = (x'/w', y'/w')
    The division by w' is what lets a single matrix represent perspective
    (not just affine) distortion - w' varies per point depending on depth,
    which is exactly the "things farther away look smaller" effect a tilted
    camera introduces. 4 point correspondences give 8 equations (2 per
    point: one for x, one for y), which exactly determines the 8 unknowns
    in H - that's why exactly 4 markers, not 3 or 5, are needed here.

    warpPerspective then applies H^-1 for every pixel in the OUTPUT canvas
    to find where to sample FROM in the input image (inverse warping -
    this avoids gaps in the output that forward-mapping would leave).
    INTER_CUBIC interpolates using a weighted neighborhood of 16 nearby
    pixels rather than 1 (nearest) or 4 (bilinear), giving smoother edges
    on the survivor blobs we're about to threshold.
    """
    dst_pts = np.array([
        [0, 0], [CANVAS_SIZE, 0],
        [CANVAS_SIZE, CANVAS_SIZE], [0, CANVAS_SIZE],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (CANVAS_SIZE, CANVAS_SIZE), flags=cv2.INTER_CUBIC)


def build_grid():
    """
    Once the canvas is a known, fixed 900x900 square, the grid doesn't
    need to be detected from pixels at all - it can be computed directly.

    Math: dividing the canvas into GRID_CELLS (12) equal cells means each
    cell is CANVAS_SIZE / GRID_CELLS = 900 / 12 = 75 px wide. The interior
    GRID LINES sit at multiples of that cell size: 75, 150, ..., 825 -
    that's 11 lines (GRID_CELLS - 1), since a 12-cell grid has 13 total
    lines but only 11 are "interior" (excluding the two outer edges).
    np.linspace(75, 825, 11) generates exactly these 11 evenly spaced
    values - spacing = (825 - 75) / (11 - 1) = 75, confirming it lines up
    with the cell size.

    The 121 intersections are the Cartesian product of the 11 x-values and
    11 y-values (11 x 11 = 121). Listed here in row-major order (y varies
    slowest, x fastest), so flat index i maps back to (row, col) via
    row, col = divmod(i, 11) - this ordering has to match nearest_labels()
    below, since that's where the reverse mapping happens.
    """
    cell = CANVAS_SIZE / GRID_CELLS
    coords = np.linspace(cell, CANVAS_SIZE - cell, LINES)  # 11 interior lines
    return np.array([[x, y] for y in coords for x in coords])  # (121, 2)


def white_balance_gray_world(bgr):
    """
    Gray-world white balance - corrects a GLOBAL color cast caused by the
    light source itself (tungsten skews warm/orange, fluorescent skews
    green, overcast daylight skews blue). This is a different problem
    from uneven brightness: two photos of the same arena under different
    lights can have identical exposure but noticeably different color,
    which would shift where "red" and "yellow" land in LAB space.

    Math: assumes that, averaged over the whole scene, the true reflected
    color should be neutral gray - i.e. the mean of B, G, and R should be
    roughly equal. If one channel's mean is off (e.g. blue reads low
    under warm light), that gap is attributed to the light source, not
    the scene, and corrected by scaling each channel so all three means
    converge on their shared average:
        target = (mean(B) + mean(G) + mean(R)) / 3
        channel_corrected = channel * (target / mean(channel))
    This preserves relative color differences (red vs. yellow vs.
    background) while removing the overall cast.
    """
    b, g, r = cv2.split(bgr.astype(np.float32))
    b_avg, g_avg, r_avg = b.mean(), g.mean(), r.mean()
    gray_avg = (b_avg + g_avg + r_avg) / 3.0
    b = np.clip(b * (gray_avg / b_avg), 0, 255)
    g = np.clip(g * (gray_avg / g_avg), 0, 255)
    r = np.clip(r * (gray_avg / r_avg), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def normalize_lighting(bgr):
    """
    Two normalization passes, fixing two DIFFERENT problems, in this
    specific order:
      1. White balance (gray-world) - corrects the photo's overall color
         cast from the light source, before any color channel is trusted.
      2. CLAHE on L, in LAB - corrects LOCAL brightness variation (shadow
         vs. glare within the same photo) without touching color.
    Order matters: CLAHE only ever touches L, so it can't undo step 1,
    but running white balance AFTER CLAHE would let per-tile brightness
    changes bias the channel means white balance depends on. White
    balance first means the a/b channels handed to segmentation reflect
    the survivors' true color as closely as the input allows, regardless
    of which light they were photographed under.

    Returns the split L, a, b channels directly (not remerged to BGR) -
    segmentation below reads a/b directly, so there's no need to convert
    back and forth.
    """
    balanced = white_balance_gray_world(bgr)
    l, a, b = cv2.split(cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return l, a, b


def segment(rectified, channel, low, high):
    """
    Color segmentation directly on a LAB color channel, cleaned up with
    morphology.

    Why LAB's a/b instead of HSV's hue: a and b are Cartesian (0 = fully
    green/blue, 128 = neutral gray, 255 = fully red/yellow), not circular
    like hue - so there's no wraparound and no need to OR two ranges
    together the way red does in HSV. Red survivors read as high `a`
    with `b` near neutral; yellow survivors read as high `b` with `a`
    near neutral - genuinely different axes, rather than two nearby
    slices of one hue wheel, which also makes red/yellow easier to tell
    apart from each other.

    Morphology (set operations on the binary mask, using a 5x5 elliptical
    structuring element as the neighborhood):
      - OPEN (erode then dilate): erosion shrinks every white region by
        "peeling off" pixels that don't have a full 5x5 neighborhood of
        white around them - this deletes small noise specks entirely,
        since they're smaller than the kernel. Dilation then grows the
        surviving regions back to roughly their original size.
      - CLOSE (dilate then erode): the reverse order - first fills in
        small black gaps/holes inside a blob (dilation), then shrinks
        back (erosion), without deleting the blob itself.
    Net effect: small noise disappears, small holes inside real survivor
    blobs get filled, real blobs keep roughly their true size and shape.
    """
    _, a, b = normalize_lighting(rectified)
    chan = {"a": a, "b": b}[channel]
    mask = cv2.inRange(chan, low, high)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def centroids_from_mask(mask, min_area_frac=0.0005):
    """
    Finds each blob's contour, filters noise by area, and computes the
    true centroid (center of mass) of each remaining blob.

    Area filter: min_area is expressed as a FRACTION of the canvas area
    (900*900) rather than a fixed pixel count, so the same threshold stays
    meaningful regardless of image resolution.

    Centroid math via image moments: for a binary region, the raw moments
    are:
        m00 = sum of all pixel values in the region  (= pixel COUNT, since
              each foreground pixel = 1)  -> this is the region's AREA
        m10 = sum of (x * pixel_value) over the region
        m01 = sum of (y * pixel_value) over the region
    The centroid is then:
        cx = m10 / m00,   cy = m01 / m00
    which is exactly the definition of center of mass: the average x and
    average y position, weighted by "mass" (here, uniform mass = 1 per
    foreground pixel). This is more correct than a bounding-box center for
    any non-symmetric shape (e.g. a triangle) - a triangle's true centroid
    sits at 1/3 of its height from the base, not at the midpoint of its
    bounding box.
    """
    min_area = min_area_frac * CANVAS_SIZE * CANVAS_SIZE
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:          # degenerate contour (a line/point, zero area)
            continue                # - guard against division by zero below
        points.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    return points


def nearest_labels(points, grid):
    """
    For each survivor centroid, finds the closest of the 121 precomputed
    grid intersections and converts that to a label like "C6".

    Math: `grid - np.array([cx, cy])` uses NumPy broadcasting to subtract
    the single point (cx, cy) from all 121 grid rows AT ONCE, giving a
    (121, 2) array of (dx, dy) differences. np.linalg.norm(..., axis=1)
    then computes, for each of the 121 rows independently:
        distance = sqrt(dx^2 + dy^2)
    which is the standard Euclidean distance. argmin picks the index of
    the smallest of the 121 distances - i.e. brute-force nearest-neighbor
    search. With only 121 candidate points this is cheap even done this
    way (no need for a k-d tree or similar).

    Converting the flat index back to a label: build_grid() laid the 121
    points out in row-major order (y outer loop, x inner loop), so
    divmod(idx, 11) recovers (row, col) exactly as they were generated.
    Column -> letter uses the fact that chr(65) is 'A' in ASCII, so
    chr(65 + col) walks A, B, C, ... as col goes 0, 1, 2, ...; the row
    number is 1-indexed (row + 1) to match a human-readable grid label
    instead of a 0-indexed array position.
    """
    labels = []
    for (cx, cy) in points:
        dists = np.linalg.norm(grid - np.array([cx, cy]), axis=1)  # (121,) distances
        idx = int(np.argmin(dists))
        row, col = divmod(idx, LINES)
        labels.append(f"{chr(65 + col)}{row + 1}")
    return labels


def write_results(path, marker_ids, critical, stable):
    """Writes the exact 4-line format the evaluator expects. Order within
    each comma-separated list doesn't matter - the evaluator compares the
    lists as sets, not as ordered sequences."""
    with open(path, "w") as f:
        f.write(f"Detected marker IDs: {sorted(marker_ids)}\n")
        f.write("\n")
        f.write(f"Critical Survivors: {', '.join(critical)}\n")
        f.write(f"Stable Survivors: {', '.join(stable)}\n")


def main():
    parser = argparse.ArgumentParser(description="Task 1A - Khoj-o-Drone")
    parser.add_argument("--image", required=True, help="path to the arena image")
    args = parser.parse_args()

    # 1. Load + validate the input image exists and decoded correctly.
    img = load_image(args.image)

    # 2. Find the 4 ArUco markers; exit loudly if any of the required IDs
    #    is missing rather than silently producing a wrong homography.
    markers = detect_markers(img)

    # 3. Work out which marker corner is TL/TR/BR/BL from geometry alone.
    src_pts = order_corners(markers)

    # 4. Warp the tilted arena photo into a flat, fixed-size top-down view.
    rectified = rectify(img, src_pts)

    # 5. Compute the 121 grid-intersection coordinates on that fixed canvas.
    grid = build_grid()

    # 6. Segment survivors by color, then reduce each blob to one point.
    #    White balance + CLAHE (inside segment -> normalize_lighting) run
    #    first, so these LAB thresholds see color corrected for both the
    #    light source's cast and local brightness variation.
    red_mask = segment(rectified, "a", 150, 255)      # high a = red
    yellow_mask = segment(rectified, "b", 150, 255)   # high b = yellow

    # 7. Snap each survivor's centroid to its nearest grid intersection.
    critical = nearest_labels(centroids_from_mask(red_mask), grid)
    stable = nearest_labels(centroids_from_mask(yellow_mask), grid)

    # 8. Write the results file next to the input image.
    out_path = f"{args.image.rsplit('.', 1)[0]}_results.txt"
    write_results(out_path, markers.keys(), critical, stable)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
