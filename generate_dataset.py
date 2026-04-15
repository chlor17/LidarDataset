"""
Synthetic LiDAR dataset generator — writes valid LAS 1.2 files (Point Format 0).
Produces realistic urban/mixed-terrain scenes with ASPRS standard classes:
  2 = Ground, 3 = Low Veg, 4 = Med Veg, 5 = High Veg, 6 = Building, 9 = Water
"""

import struct
import numpy as np
import os
import math

# ── ASPRS class config ──────────────────────────────────────────────────────
CLASSES = {
    "ground":    2,
    "low_veg":   3,
    "med_veg":   4,
    "high_veg":  5,
    "building":  6,
    "water":     9,
}

CLASS_NAMES = {v: k for k, v in CLASSES.items()}

# ── LAS 1.2 binary writer ───────────────────────────────────────────────────

def write_las(path, x, y, z, classification, intensity=None):
    """Write arrays to a LAS 1.2 file with Point Format 0."""
    n = len(x)
    assert len(y) == n and len(z) == n and len(classification) == n

    if intensity is None:
        intensity = np.random.randint(100, 4000, size=n, dtype=np.uint16)

    scale = 0.001
    x_offset = float(np.mean(x))
    y_offset = float(np.mean(y))
    z_offset = 0.0

    xi = np.round((x - x_offset) / scale).astype(np.int32)
    yi = np.round((y - y_offset) / scale).astype(np.int32)
    zi = np.round((z - z_offset) / scale).astype(np.int32)

    header_size = 227
    point_record_length = 20
    offset_to_points = header_size  # no VLRs

    with open(path, "wb") as f:
        # ── public header block ──────────────────────────────────────────
        f.write(b"LASF")                          # file signature
        f.write(struct.pack("<H", 0))              # file source ID
        f.write(struct.pack("<H", 0))              # global encoding
        f.write(struct.pack("<I", 0))              # GUID data 1
        f.write(struct.pack("<H", 0))              # GUID data 2
        f.write(struct.pack("<H", 0))              # GUID data 3
        f.write(b"\x00" * 8)                       # GUID data 4
        f.write(struct.pack("<B", 1))              # version major
        f.write(struct.pack("<B", 2))              # version minor
        f.write(b"SyntheticLiDAR\x00" + b"\x00" * 18)  # system identifier (32)
        f.write(b"LidarDatasetGen\x00" + b"\x00" * 17) # generating software (32)
        f.write(struct.pack("<H", 1))              # file creation day
        f.write(struct.pack("<H", 2026))           # file creation year
        f.write(struct.pack("<H", header_size))    # header size
        f.write(struct.pack("<I", offset_to_points))
        f.write(struct.pack("<I", 0))              # number of VLRs
        f.write(struct.pack("<B", 0))              # point data format ID = 0
        f.write(struct.pack("<H", point_record_length))
        f.write(struct.pack("<I", n))              # number of point records

        # number of points by return (5 values)
        f.write(struct.pack("<I", n))
        for _ in range(4):
            f.write(struct.pack("<I", 0))

        # scale factors
        f.write(struct.pack("<d", scale))          # X scale
        f.write(struct.pack("<d", scale))          # Y scale
        f.write(struct.pack("<d", scale))          # Z scale

        # offsets
        f.write(struct.pack("<d", x_offset))
        f.write(struct.pack("<d", y_offset))
        f.write(struct.pack("<d", z_offset))

        # bounding box (max/min pairs: X, Y, Z)
        f.write(struct.pack("<d", float(np.max(x))))
        f.write(struct.pack("<d", float(np.min(x))))
        f.write(struct.pack("<d", float(np.max(y))))
        f.write(struct.pack("<d", float(np.min(y))))
        f.write(struct.pack("<d", float(np.max(z))))
        f.write(struct.pack("<d", float(np.min(z))))

        # ── point data records (Format 0 = 20 bytes each) ────────────────
        for i in range(n):
            f.write(struct.pack("<i", int(xi[i])))
            f.write(struct.pack("<i", int(yi[i])))
            f.write(struct.pack("<i", int(zi[i])))
            f.write(struct.pack("<H", int(intensity[i])))
            f.write(struct.pack("<B", 0b00010001))  # return 1 of 1
            f.write(struct.pack("<B", int(classification[i])))
            f.write(struct.pack("<b", 0))           # scan angle
            f.write(struct.pack("<B", 0))           # user data
            f.write(struct.pack("<H", 1))           # point source ID


# ── scene generators ────────────────────────────────────────────────────────

def ground_plane(rng, n, xlim, ylim, noise=0.05):
    x = rng.uniform(*xlim, n)
    y = rng.uniform(*ylim, n)
    z = rng.normal(0.0, noise, n)
    return x, y, z


def add_building(rng, cx, cy, width, depth, height, density=800):
    """Rectangular building: flat roof + 4 facades."""
    pts = []
    # roof
    n_roof = int(width * depth * density * 0.3)
    rx = rng.uniform(cx - width / 2, cx + width / 2, n_roof)
    ry = rng.uniform(cy - depth / 2, cy + depth / 2, n_roof)
    rz = rng.normal(height, 0.03, n_roof)
    pts.append((rx, ry, rz))
    # facades (4 sides)
    for _ in range(4):
        n_wall = int(max(width, depth) * height * density * 0.15)
        if _ < 2:
            wx = np.full(n_wall, cx + (1 if _ == 0 else -1) * width / 2)
            wy = rng.uniform(cy - depth / 2, cy + depth / 2, n_wall)
        else:
            wx = rng.uniform(cx - width / 2, cx + width / 2, n_wall)
            wy = np.full(n_wall, cy + (1 if _ == 2 else -1) * depth / 2)
        wz = rng.uniform(0, height, n_wall) + rng.normal(0, 0.02, n_wall)
        pts.append((wx, wy, wz))
    x = np.concatenate([p[0] for p in pts])
    y = np.concatenate([p[1] for p in pts])
    z = np.concatenate([p[2] for p in pts])
    return x, y, z


def add_tree(rng, cx, cy, trunk_h, canopy_r, canopy_h, n_pts=1500):
    """Conical canopy + vertical trunk."""
    # canopy
    phi = rng.uniform(0, 2 * math.pi, n_pts)
    r   = rng.triangular(0, 0, canopy_r, n_pts)
    z_c = rng.uniform(trunk_h, trunk_h + canopy_h, n_pts)
    # taper radius with height
    taper = (trunk_h + canopy_h - z_c) / canopy_h
    r_eff = r * taper
    tx = cx + r_eff * np.cos(phi)
    ty = cy + r_eff * np.sin(phi)
    # trunk
    n_trunk = 80
    trunk_r = 0.15
    theta_t = rng.uniform(0, 2 * math.pi, n_trunk)
    bx = cx + trunk_r * np.cos(theta_t)
    by = cy + trunk_r * np.sin(theta_t)
    bz = rng.uniform(0, trunk_h, n_trunk)
    x = np.concatenate([tx, bx])
    y = np.concatenate([ty, by])
    z = np.concatenate([z_c, bz])
    return x, y, z


def add_water_body(rng, cx, cy, rx, ry, n=1200):
    """Elliptical flat water surface."""
    phi = rng.uniform(0, 2 * math.pi, n)
    r   = np.sqrt(rng.uniform(0, 1, n))
    x   = cx + rx * r * np.cos(phi)
    y   = cy + ry * r * np.sin(phi)
    z   = rng.normal(-0.05, 0.02, n)
    return x, y, z


# ── scene assembly ──────────────────────────────────────────────────────────

def generate_scene(rng, scene_id, tile_size=100.0, n_ground=60000):
    xlim = (0.0, tile_size)
    ylim = (0.0, tile_size)

    all_x, all_y, all_z, all_cls = [], [], [], []

    # ground
    gx, gy, gz = ground_plane(rng, n_ground, xlim, ylim)
    all_x.append(gx); all_y.append(gy); all_z.append(gz)
    all_cls.append(np.full(len(gx), CLASSES["ground"], dtype=np.uint8))

    # buildings (3-6 per tile)
    n_buildings = rng.integers(3, 7)
    for _ in range(n_buildings):
        cx = rng.uniform(15, tile_size - 15)
        cy = rng.uniform(15, tile_size - 15)
        w  = rng.uniform(8, 20)
        d  = rng.uniform(8, 20)
        h  = rng.uniform(5, 25)
        bx, by, bz = add_building(rng, cx, cy, w, d, h)
        all_x.append(bx); all_y.append(by); all_z.append(bz)
        all_cls.append(np.full(len(bx), CLASSES["building"], dtype=np.uint8))

    # trees (10-25 per tile, mixed height = low/med/high veg)
    n_trees = rng.integers(10, 26)
    for _ in range(n_trees):
        cx = rng.uniform(5, tile_size - 5)
        cy = rng.uniform(5, tile_size - 5)
        trunk_h = rng.uniform(1, 4)
        canopy_r = rng.uniform(1.5, 5)
        canopy_h = rng.uniform(2, 10)
        total_h  = trunk_h + canopy_h
        cls_val  = CLASSES["low_veg"] if total_h < 4 else (
                   CLASSES["med_veg"] if total_h < 10 else CLASSES["high_veg"])
        tx, ty, tz = add_tree(rng, cx, cy, trunk_h, canopy_r, canopy_h)
        all_x.append(tx); all_y.append(ty); all_z.append(tz)
        all_cls.append(np.full(len(tx), cls_val, dtype=np.uint8))

    # water (0-1 body per tile)
    if rng.random() > 0.4:
        cx = rng.uniform(20, tile_size - 20)
        cy = rng.uniform(20, tile_size - 20)
        rx = rng.uniform(5, 15)
        ry = rng.uniform(5, 15)
        wx, wy, wz = add_water_body(rng, cx, cy, rx, ry)
        all_x.append(wx); all_y.append(wy); all_z.append(wz)
        all_cls.append(np.full(len(wx), CLASSES["water"], dtype=np.uint8))

    x   = np.concatenate(all_x).astype(np.float64)
    y   = np.concatenate(all_y).astype(np.float64)
    z   = np.concatenate(all_z).astype(np.float64)
    cls = np.concatenate(all_cls)

    # add scan noise & occlusion-style intensity variation
    intensity = (500 + 3000 * np.exp(-0.05 * z)).astype(np.uint16)
    intensity = np.clip(intensity + rng.integers(-200, 200, len(x)), 0, 65535).astype(np.uint16)

    return x, y, z, cls, intensity


# ── main ────────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "tiles")
    os.makedirs(out_dir, exist_ok=True)

    n_tiles   = 20
    rng       = np.random.default_rng(42)

    class_counts_total = {v: 0 for v in CLASSES.values()}

    print(f"Generating {n_tiles} LAS tiles → {out_dir}/")
    for i in range(n_tiles):
        # offset each tile on a grid so coordinates don't overlap
        col, row = i % 5, i // 5
        x, y, z, cls, intensity = generate_scene(rng, i)
        x += col * 110.0
        y += row * 110.0

        for c in np.unique(cls):
            class_counts_total[int(c)] = class_counts_total.get(int(c), 0) + int(np.sum(cls == c))

        fname = os.path.join(out_dir, f"tile_{i:03d}.las")
        write_las(fname, x, y, z, cls, intensity)
        n_pts = len(x)
        size_kb = os.path.getsize(fname) / 1024
        print(f"  tile_{i:03d}.las  {n_pts:>8,} pts  {size_kb:>8.1f} KB")

    print("\nClass distribution across all tiles:")
    total = sum(class_counts_total.values())
    for cls_id, count in sorted(class_counts_total.items()):
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        print(f"  {cls_id:2d}  {name:<12}  {count:>9,}  ({100*count/total:.1f}%)")
    print(f"\nTotal points: {total:,}")
    print("Done.")


if __name__ == "__main__":
    main()
