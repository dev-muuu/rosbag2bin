# rosbag2bin

Converts `sensor_msgs/msg/PointCloud2` messages in ROS bags to KITTI binary
format — one `.bin` file per frame, each an `Nx4` `float32` array of
`(x, y, z, intensity)`.

## Requirements

```bash
pip install rosbags numpy
```

Python 3.10 or newer. No ROS installation needed.

## Usage

```bash
python3 rosbag2bin.py
```

The same command handles ROS2 bag directories (sqlite3 and mcap) and ROS1
`.bag` files — the script detects which is which, so nothing needs to change
between them.

`bag_root` may be a single bag or a directory searched recursively, in which
case every bag below it is converted.

### Configuring in code

Defaults live in the `CONFIG` block at the top of the script:

```python
# A single bag, or a directory to search recursively for bags.
BAG_ROOT = '/data_1/evaluate_log/raw'

# Each bag lands in its own directory here: OUTPUT_ROOT/<bag name>/
OUTPUT_ROOT = '/data_1/kitti'

# PointCloud2 topic to extract. Bags that lack it are reported and skipped.
TOPIC = '/pointcloud/vlp16'
```

## Output structure

Each bag found under `bag_root` gets its own directory under `output_root`,
named after the bag.

Given this input tree:

```
/data_1/evaluate_log/raw/        <- bag_root
├── test1/
│   ├── metadata.yaml
│   └── test1_0.db3
└── test2/
    ├── metadata.yaml
    └── test2_0.db3
```

the converter produces:

```
/data_1/kitti/                   <- output_root
├── test1/
│   ├── 000000_1775094601779.bin
│   ├── 000001_1775094602290.bin
│   └── ...
└── test2/
    ├── 000000_1775094724786.bin
    ├── ...
    └── skipped.txt
```

### File names

```
{frame index:06d}_{header stamp in ms}.bin
```

Frame indices are contiguous. Frames with no points are skipped rather than
written as empty files; when any are skipped, a `skipped.txt` lists the source
frames that were left out:

```
# source frames skipped because they had no points
# source_index	stamp_ms
0	1786517343987
1	1786517344035
```

### File contents

A flat `float32` array of `N` points by 4 columns — the KITTI layout:

| Column | Value |
|---|---|
| 0 | `x` |
| 1 | `y` |
| 2 | `z` |
| 3 | `intensity` (0 when the cloud has no intensity field) |

