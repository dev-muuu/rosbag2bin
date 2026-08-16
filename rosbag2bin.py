#!/usr/bin/env python3
"""
ROS bag to KITTI binary converter.

Reads PointCloud2 messages and writes one KITTI .bin per frame -- an Nx4
float32 array of (x, y, z, intensity).

Handles ROS2 bag directories (sqlite3 and mcap, any distro) and ROS1 .bag
files alike, and needs no ROS installation -- only `pip install rosbags numpy`.

BAG_ROOT may be a single bag or a directory containing many; every bag found
below it is converted into its own directory under OUTPUT_ROOT:

    OUTPUT_ROOT/<bag name>/000000_<stamp_ms>.bin

Edit the defaults in the CONFIG block below, or override them on the command
line (`--help` for the full list).
"""

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit(
        "[ERROR] 'numpy' is required but not installed.\n"
        "        Install it with:  pip install numpy")

try:
    from rosbags.highlevel import AnyReader
except ImportError:
    sys.exit(
        "[ERROR] 'rosbags' is required but not installed.\n"
        "        Install it with:  pip install rosbags\n"
        "        (pure Python, needs Python >= 3.10; no ROS installation required)\n"
        "        To keep it out of the system site-packages, use a virtualenv:\n"
        "            python3 -m venv ~/.venvs/rosbags\n"
        "            ~/.venvs/rosbags/bin/pip install rosbags\n"
        "            ~/.venvs/rosbags/bin/python {}".format(sys.argv[0]))

try:
    # rosbags >= 0.10
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    Stores = get_typestore = None


# ===========================================================================
# CONFIG -- edit these defaults; every one can also be overridden on the CLI.
# ===========================================================================

# A single bag, or a directory to search recursively for bags.
BAG_ROOT = ''

# Each bag lands in its own directory here: OUTPUT_ROOT/<bag name>/
OUTPUT_ROOT = ''

# PointCloud2 topic to extract. Bags that lack it are reported and skipped.
TOPIC = '/pointcloud/vlp16'

# ===========================================================================


# Message definitions used to deserialize ROS2 bags, which -- unlike ROS1 .bag
# files -- do not embed their own. Not a knob: sensor_msgs/msg/PointCloud2 is
# identical across every ROS2 distro, so each typestore yields byte-identical
# output. Only a bag carrying custom message types would care.
STORE = 'ROS2_HUMBLE'

# sensor_msgs/PointField datatype -> numpy base type string
TYPE_MAP = {
    1: 'i1',  # INT8
    2: 'u1',  # UINT8
    3: 'i2',  # INT16
    4: 'u2',  # UINT16
    5: 'i4',  # INT32
    6: 'u4',  # UINT32
    7: 'f4',  # FLOAT32
    8: 'f8',  # FLOAT64
}

REQUIRED_FIELDS = ('x', 'y', 'z')


def build_point_dtype(msg):
    """
    Build a numpy structured dtype mirroring the exact memory layout of a
    PointCloud2 message.

    Honours field offsets, point_step (as the stride) and is_bigendian, so
    clouds with inter-field or trailing padding -- e.g. PCL's PointXYZI, which
    pads x/y/z out to 16 bytes and uses point_step=32 -- are read correctly.
    """
    endian = '>' if msg.is_bigendian else '<'

    names, formats, offsets = [], [], []
    for field in msg.fields:
        if not field.name:
            # Unnamed padding field; it is implied by the neighbouring offsets,
            # so there is nothing to read.
            continue

        base = TYPE_MAP.get(field.datatype)
        if base is None:
            raise ValueError(
                "field '{}' has unsupported PointField datatype {}".format(
                    field.name, field.datatype))

        if field.name in names:
            raise ValueError("duplicate field name '{}' in PointCloud2".format(field.name))

        # count > 1 means the field is a fixed-size array (e.g. a normal vector)
        count = getattr(field, 'count', 1) or 1

        names.append(field.name)
        formats.append((endian + base, count) if count > 1 else endian + base)
        offsets.append(field.offset)

    if not names:
        raise ValueError('PointCloud2 message declares no readable fields')

    return np.dtype({
        'names': names,
        'formats': formats,
        'offsets': offsets,
        'itemsize': msg.point_step,
    })


def pointcloud2_to_array(msg, dtype):
    """Decode a PointCloud2 message into a structured array, one record per point."""
    width, height = msg.width, msg.height
    point_step, row_step = msg.point_step, msg.row_step
    available = len(msg.data)

    # Organized clouds may pad each row, in which case row_step exceeds the
    # bytes actually occupied by the row's points and we must read row by row.
    if height > 1 and row_step != width * point_step:
        rows = []
        for r in range(height):
            offset = r * row_step
            n = min(width, max(0, (available - offset) // point_step))
            if n == 0:
                continue
            rows.append(np.frombuffer(msg.data, dtype=dtype, count=n, offset=offset))
        if not rows:
            return np.empty(0, dtype=dtype)
        return np.concatenate(rows)

    n = min(width * height, available // point_step)
    if n == 0:
        return np.empty(0, dtype=dtype)
    return np.frombuffer(msg.data, dtype=dtype, count=n, offset=0)


def drop_non_finite(points):
    """Remove points whose x/y/z are NaN or infinite."""
    mask = (np.isfinite(points['x'])
            & np.isfinite(points['y'])
            & np.isfinite(points['z']))
    return points[mask]


def save_pointcloud_as_bin(points, output_path):
    """Save point cloud as KITTI binary format (x, y, z, intensity), float32."""
    data = np.zeros((len(points), 4), dtype=np.float32)
    data[:, 0] = points['x']
    data[:, 1] = points['y']
    data[:, 2] = points['z']
    if 'intensity' in points.dtype.names:
        data[:, 3] = points['intensity']
    # else: leave the intensity column at 0

    data.tofile(output_path)


def stamp_to_ms(msg):
    """Header stamp in milliseconds, tolerating ROS1-style field names."""
    stamp = msg.header.stamp
    sec = getattr(stamp, 'sec', None)
    if sec is None:
        sec = stamp.secs
    nanosec = getattr(stamp, 'nanosec', None)
    if nanosec is None:
        nanosec = stamp.nsecs
    return sec * 1000 + nanosec // 1000000


def describe_layout(msg, dtype):
    parts = ['{}@{}:{}'.format(name, dtype.fields[name][1], dtype.fields[name][0].str)
             for name in dtype.names]
    return '{}x{} point_step={} row_step={} is_dense={} [{}]'.format(
        msg.width, msg.height, msg.point_step, msg.row_step,
        msg.is_dense, ', '.join(parts))


def open_reader(bag_path, store=STORE):
    """
    Open a bag with AnyReader.

    rosbag2 bags do not embed message definitions, so rosbags >= 0.10 needs an
    explicit typestore to deserialize them; without it AnyReader raises
    "Bag contains no type definitions".
    """
    if get_typestore is None:
        return AnyReader([bag_path])  # rosbags < 0.10 carries its own types

    if not hasattr(Stores, store):
        raise ValueError('unknown typestore {!r}; available: {}'.format(
            store, ', '.join(s.name for s in Stores)))
    return AnyReader([bag_path], default_typestore=get_typestore(getattr(Stores, store)))


def is_bag(path):
    """A ROS2 bag directory (holds metadata.yaml) or a ROS1 .bag file."""
    path = Path(path)
    if path.is_file():
        return path.suffix == '.bag'
    return path.is_dir() and (path / 'metadata.yaml').is_file()


def find_bags(root):
    """
    Every bag at or below root, sorted.

    Returns [root] when root is itself a bag, so the same entry point handles
    both a single bag and a tree of them.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError('bag root does not exist: {}'.format(root))
    if is_bag(root):
        return [root]

    bags = {md.parent for md in root.rglob('metadata.yaml')}
    bags |= {p for p in root.rglob('*.bag') if p.is_file()}
    return sorted(bags)


def output_dir_for(bag, bag_root, output_root, taken):
    """
    OUTPUT_ROOT/<bag name>/ -- falling back to a path-qualified name when two
    bags in the tree share a name, so neither silently overwrites the other.
    """
    bag, bag_root, output_root = Path(bag), Path(bag_root), Path(output_root)
    name = bag.stem if bag.is_file() else bag.name

    if name in taken:
        try:
            rel = bag.relative_to(bag_root if bag_root.is_dir() else bag_root.parent)
        except ValueError:
            rel = Path(name)
        qualified = '_'.join(rel.with_suffix('').parts)
        print('[WARN] Duplicate bag name {!r}; using {!r} instead'.format(name, qualified))
        name = qualified

    taken.add(name)
    return output_root / name


def rosbag_to_bin(bag_path, topic_name, output_dir, filter_nans=True, store=STORE):
    """Convert a bag's PointCloud2 topic to per-frame binary files."""
    bag_path = Path(bag_path)
    output_dir = Path(output_dir)

    print('[ROSBAG2BIN] Reading bag: {}'.format(bag_path))
    print('[ROSBAG2BIN] Topic: {}'.format(topic_name))
    print('[ROSBAG2BIN] Output: {}'.format(output_dir))

    frame_count = 0      # index of the next file written
    source_index = 0     # index of the message in the bag, empty frames included
    dropped_total = 0
    skipped = []
    dtype = None

    try:
        # AnyReader handles both ROS1 .bag files and ROS2 bag directories.
        with open_reader(bag_path, store) as reader:
            connections = [c for c in reader.connections if c.topic == topic_name]

            if not connections:
                print("[ERROR] Topic '{}' not found in bag".format(topic_name))
                print('[INFO] Available topics:')
                for c in sorted(reader.connections, key=lambda c: c.topic):
                    print('  - {} ({})'.format(c.topic, c.msgtype))
                return False

            print('[ROSBAG2BIN] Found {} connection(s) for topic {}'.format(
                len(connections), topic_name))

            # Created only once the bag and topic are known to be usable, so a
            # failed bag leaves no empty directory behind.
            output_dir.mkdir(parents=True, exist_ok=True)

            for connection, _, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)

                # The layout is fixed for a topic, so derive and validate it once.
                if dtype is None:
                    dtype = build_point_dtype(msg)
                    missing = [f for f in REQUIRED_FIELDS if f not in dtype.names]
                    if missing:
                        print('[ERROR] PointCloud2 is missing required field(s): {}'.format(
                            ', '.join(missing)))
                        print('[INFO] Fields present: {}'.format(', '.join(dtype.names)))
                        return False
                    if 'intensity' not in dtype.names:
                        print('[WARN] No intensity field; that column will be written as 0')
                    if msg.height > 1 and msg.row_step != msg.width * msg.point_step:
                        print('[WARN] Organized cloud with padded rows; reading row by row')
                    print('[ROSBAG2BIN] Layout: {}'.format(describe_layout(msg, dtype)))

                points = pointcloud2_to_array(msg, dtype)

                if filter_nans and not msg.is_dense:
                    before = len(points)
                    points = drop_non_finite(points)
                    dropped_total += before - len(points)

                # Empty frames are skipped so the output numbering stays
                # contiguous; skipped.txt records which source frames were lost.
                if len(points) == 0:
                    skipped.append((source_index, stamp_to_ms(msg)))
                    source_index += 1
                    continue

                filename = output_dir / '{:06d}_{}.bin'.format(frame_count, stamp_to_ms(msg))
                save_pointcloud_as_bin(points, filename)

                frame_count += 1
                source_index += 1

                if frame_count % 100 == 0:
                    print('[ROSBAG2BIN] Processed {} frames...'.format(frame_count))

    except Exception as e:
        print('[ERROR] Exception in rosbag_to_bin: {}'.format(e))
        import traceback
        traceback.print_exc()
        return False

    if dropped_total:
        print('[ROSBAG2BIN] Dropped {} non-finite point(s)'.format(dropped_total))

    if skipped:
        report = output_dir / 'skipped.txt'
        with open(report, 'w') as fh:
            fh.write('# source frames skipped because they had no points\n')
            fh.write('# source_index\tstamp_ms\n')
            for idx, ms in skipped:
                fh.write('{}\t{}\n'.format(idx, ms))
        print('[ROSBAG2BIN] Skipped {} empty frame(s); see {}'.format(len(skipped), report))

    print('[ROSBAG2BIN] Successfully converted {} frames'.format(frame_count))
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert PointCloud2 messages in ROS1/ROS2 bags to KITTI .bin files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Defaults come from the CONFIG block at the top of this file:\n'
               '  BAG_ROOT    = {}\n'
               '  OUTPUT_ROOT = {}\n'
               '  TOPIC       = {}\n'.format(BAG_ROOT, OUTPUT_ROOT, TOPIC))
    parser.add_argument('bag_root', nargs='?', default=BAG_ROOT,
                        help='A bag, or a directory searched recursively for bags '
                             '(default: BAG_ROOT)')
    parser.add_argument('output_root', nargs='?', default=OUTPUT_ROOT,
                        help='Each bag is written to <output_root>/<bag name>/ '
                             '(default: OUTPUT_ROOT)')
    parser.add_argument('-t', '--topic', default=TOPIC,
                        help='PointCloud2 topic to extract (default: {})'.format(TOPIC))
    parser.add_argument('-n', '--limit', type=int, metavar='N',
                        help='Convert only the first N bags found (for a test run)')
    parser.add_argument('--dry-run', action='store_true',
                        help='List the bags and their output directories, convert nothing')
    parser.add_argument('--no-filter-nans', dest='filter_nans', action='store_false',
                        help='Keep non-finite points instead of dropping them '
                             'when the cloud is not dense')

    args = parser.parse_args()

    # Path('') silently means the current directory, so an unset root would
    # scan the wrong tree or write output somewhere unintended without error.
    for value, const, arg in ((args.bag_root, 'BAG_ROOT', 'bag_root'),
                              (args.output_root, 'OUTPUT_ROOT', 'output_root')):
        if not str(value).strip():
            sys.exit('[ERROR] {} is empty; set it in the CONFIG block at the top '
                     'of this file, or pass {} on the command line'.format(const, arg))

    try:
        bags = find_bags(args.bag_root)
    except FileNotFoundError as e:
        sys.exit('[ERROR] {}'.format(e))

    if not bags:
        sys.exit('[ERROR] No bags found under {}'.format(args.bag_root))

    total_found = len(bags)
    if args.limit is not None:
        bags = bags[:args.limit]

    print('[ROSBAG2BIN] Bag root:    {}'.format(args.bag_root))
    print('[ROSBAG2BIN] Output root: {}'.format(args.output_root))
    print('[ROSBAG2BIN] Bags: {}{}'.format(
        len(bags),
        ' of {} found (--limit {})'.format(total_found, args.limit)
        if len(bags) != total_found else ''))

    taken = set()
    jobs = [(bag, output_dir_for(bag, args.bag_root, args.output_root, taken))
            for bag in bags]

    if args.dry_run:
        print('\n[DRY RUN] Nothing will be written.')
        for bag, out in jobs:
            print('  {}\n    -> {}'.format(bag, out))
        return

    ok, failed = [], []
    for i, (bag, out) in enumerate(jobs, 1):
        print('\n===== [{}/{}] {} ====='.format(i, len(jobs), bag))
        if rosbag_to_bin(bag, args.topic, out, filter_nans=args.filter_nans):
            ok.append(bag)
        else:
            failed.append(bag)

    print('\n[ROSBAG2BIN] Done: {} converted, {} failed'.format(len(ok), len(failed)))
    for bag in failed:
        print('  FAILED: {}'.format(bag))
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
