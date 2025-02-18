from random import random

import os
import numpy as np
import torch
import random
import hashlib
import matplotlib.pyplot as plt
from concurrent.futures import Future
from collections import deque
from typing import *

T = TypeVar('T')

NODEID_TO_RGB = {
    0: (160, 160, 160), # default:stone, grey
    1: (160, 160, 160), # default:cobble, grey
    4: (35, 41, 35), # default:mossycobble, grey-green
    21: (102, 51, 0), # default:dirt, brown
    22: (0, 153, 0), # default:dirt_with_grass, green
    25: (255, 255, 255), # default:dirt_with_snow, white
    33: (204, 204, 0), # default:sand, sand yellow
    35: (204, 204, 0), # default:silver_sand, sand yellow
    36: (160, 160, 160), # default:gravel, grey
    38: (255, 255, 255), # default:snow, white
    42: (25, 51, 0),  # mushroom_brown, dark green
    43: (25, 51, 0), # mushroom_red, dark green
    53: (51, 25, 0), # default:pine_tree, dark brown
    55: (25, 51, 0), # default:pine_needles, dark green
    65: (0, 0, 0), # default:stone_with_coal, black
    68: (192, 192, 192), # default:steelblock, light grey
    79: (102, 255, 255), # diamondblock, cyan
    85: (0, 153, 0), # default:grass1, green
    86: (0, 153, 0), # default:grass2, green
    118: (255, 128, 0), # coral (goal), orange
    122: (0, 0, 255), # default:river_water_source, blue
    123: (0, 0, 255), # default:river_water_flowing, blue
    126: 'hide', # air, NOCOLOR
    127: 'hide', # IGNORE, NOCOLOR
    144: (102, 255, 255), # default:glass, cyan
    161: (255, 255, 0), # default:torch_wall, yellow
    190: (160, 160, 160), # default:stair_cobble, grey
    224: (160, 160, 160), # mcl_core:stone, grey
    225: (0, 0, 0), # mcl_core:stone_with_coal, black
    226: (192, 192, 192), # mcl_core:stone_with_iron, light grey
    227: (255, 255, 0), # mcl_core:stone_with_gold, yellow
    230: (0, 0, 255), # mcl_core:stone_with_lapis, blue
    238: (64, 64, 64), # mcl_core:granite, dark grey
    264: (146, 88, 59), # mcl_core:clay, clay
    240: (224, 224, 224), # mcl_core:andesite, white
    242: (146, 88, 59), # mcl_core:diorite, clay
    244: (0, 153, 0), # mcl_core:dirt_with_grass, green
    251: (102, 51, 0), # mcl_core:dirt, brown
    253: (160, 160, 160), # mcl_core:gravel, grey
    267: (160, 160, 160), # mcl_core:cobble, grey
    268: (35, 41, 35), # mcl_core:mossycobble, grey-green
    296: (204, 0, 0), # mcl_core:lava_source, dark red
    293: (0, 0, 255), # mcl_core:water_flowing, blue
    294: (0, 0, 255), # mcl_core:water_source, blue
    307: (51, 25, 0), # mcl_core:birchtree, dark brown
    322: (0, 0, 0), # superflat:bedrock, black
    345: (25, 51, 0), # mcl_core:birchleaves, dark green
    365: (25, 51, 0), # mcl_core:vine, dark green
    388: (0, 0, 0), # xpanes:bar, black
    389: (0, 0, 0), # xpanes:bar, black
    632: (255, 255, 0),  # mcl_chests:chest_small, yellow
    633: (255, 255, 0), # mcl_chests:chest_left, yellow
    634: (255, 255, 0), # mcl_chests:chest_right, yellow
    982: (255, 255, 0), # mcl_mobspawners:spawner, yellow
    1064: (25, 51, 0), # mcl_flowers:tallgrass, dark green
    1066: (25, 51, 0), # mcl_flowers:clover, dark green
    1067: (25, 51, 0), # mcl_flowers:fourleaf_clover, dark green
    1082: (25, 51, 0), # mcl_flowers:poppy, dark green
    1084: (25, 51, 0), # mcl_flowers:dandelion, dark green
    1844: (151, 122, 107), # mcl_copper:stone_with_copper, dark clay
}

def get_file_hash(file: str) -> str:
    sha256 = hashlib.sha256()
    # Read the file from the path
    with open(file, "rb") as f:
        # Update the hash with the file content
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()

def seed_everything(seed=0):
    """
    Helper to seed random number generators.

    Note, however, that even with the seeds fixed, some non-determinism is possible.

    For more details read <https://pytorch.org/docs/stable/notes/randomness.html>.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def plot_rgb(rgb_obs):
    plt.imshow(rgb_obs)
    plt.show()

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_voxels(voxels, size=80, alpha=0.4, unknown_node_color=(255, 51, 153)):
    # Get coordinates and node IDs of all non-air voxels
    xs, ys, zs = [], [], []
    colors = []
    nodes_in_view = {}
    unknown_node_ids = []
    # Iterate through all points
    for x, y, z in np.ndindex(voxels.shape):
        node_id = voxels[x, y, z]
        if NODEID_TO_RGB.get(node_id) == 'hide':
            continue
        xs.append(x)
        ys.append(y)
        zs.append(z)
        if not node_id in NODEID_TO_RGB:
            col = unknown_node_color
            unknown_node_ids.append(node_id)
        else:
            col = NODEID_TO_RGB[node_id]
            nodes_in_view[node_id] = NODEID_TO_RGB[node_id]
        rgb = tuple(val / 255 for val in col)
        colors.append(rgb)

    print(f"Unknown node IDs: ", [int(n) for n in set(unknown_node_ids)])

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(projection='3d')
    # Make axes equal/orthonormal
    ax.set_box_aspect([1, 1, 1])
    img = ax.scatter(xs, ys, zs, c=colors, s=size, alpha=alpha)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add legend with colors
    nodes_in_view['unknown'] = unknown_node_color
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  label=f'Node {node_id}',
                                  markerfacecolor=tuple(val / 255 for val in rgb),
                                  markersize=10)
                       for node_id, rgb in nodes_in_view.items()
                       if rgb is not None]
    ax.legend(handles=legend_elements)

    set_axes_equal(ax)
    return fig, ax

class MockThreadPoolExecutor:
    """
    A single-threaded mock implementation of ThreadPoolExecutor for debugging purposes.
    Executes tasks sequentially in the main thread instead of using a thread pool.
    """

    def __init__(self, max_workers=None, thread_name_prefix='', initializer=None, initargs=()):
        self._shutdown = False
        self._tasks = deque()
        self._initializer = initializer
        self._initargs = initargs

        # Run initializer if provided
        if self._initializer:
            self._initializer(*self._initargs)

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        """
        Submit a task for execution.
        Instead of running in a separate thread, executes immediately and returns a completed Future.
        """
        if self._shutdown:
            raise RuntimeError('cannot schedule new futures after shutdown')

        future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)

        return future

    def map(self, fn: Callable[..., T], *iterables: Iterable[Any], timeout=None, chunksize=1) -> Iterable[T]:
        """
        Returns an iterator equivalent to map(fn, *iterables).
        Executes tasks sequentially instead of in parallel.
        Important: This implementation ensures all tasks are executed even if their results are not consumed,
        which is necessary for side effects like updating progress bars.
        """
        return list(self._map(fn, *iterables, timeout=timeout, chunksize=chunksize))

    def _map(self, fn: Callable[..., T], *iterables: Iterable[Any], timeout=None, chunksize=1) -> Iterable[T]:
        """
        Returns an iterator equivalent to map(fn, *iterables).
        Executes tasks sequentially instead of in parallel.
        Important: This implementation ensures all tasks are executed even if their results are not consumed,
        which is necessary for side effects like updating progress bars.
        """
        if self._shutdown:
            raise RuntimeError('cannot schedule new futures after shutdown')

        # Create a list to store futures
        futures = []

        # Submit all tasks first to ensure they all execute
        for args in zip(*iterables):
            future = self.submit(fn, *args)
            futures.append(future)

        # Now yield the results
        for future in futures:
            try:
                result = future.result(timeout=timeout)
                yield result
            except Exception as exc:
                # Continue processing even if individual tasks fail
                raise exc
                # continue

    def shutdown(self, wait=True, *, cancel_futures=False):
        """
        Signal the executor that it should free any resources.
        Since this is a mock implementation, it just sets the shutdown flag.
        """
        self._shutdown = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False
