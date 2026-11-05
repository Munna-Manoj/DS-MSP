"""Camera graph edges map source coordinates into destination coordinates."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ds_msp.rig.extrinsics import init_camera_groups
from ds_msp.rig.types import ObjectObs


def _pose(angles, translation):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    transform[:3, 3] = translation
    return transform


@pytest.mark.parametrize("reverse_observations", [False, True])
def test_sparse_camera_chain_recovers_absolute_noncommuting_poses(reverse_observations):
    # Independent absolute poses, not an expected value made by composing the edges
    # under test. Cameras 0 and 12 never observe the object together.
    expected = {
        0: np.eye(4),
        7: _pose([25, -12, 40], [0.7, -0.4, 1.2]),
        12: _pose([-35, 50, 18], [-1.3, 0.8, 0.5]),
        23: _pose([42, 9, -60], [1.1, 0.2, -0.6]),
    }
    observations = []
    for frame, pair in enumerate([(0, 7), (7, 12), (12, 23)]):
        object_pose = _pose([8 * frame, -15, 11], [0.1 * frame, 0.3, 4.0])
        for camera in pair:
            observations.append(ObjectObs(
                cam_id=camera, frame_id=frame, object_id=0,
                point_rows=np.arange(4), pts_2d=np.zeros((4, 2)),
                T_c_o=expected[camera] @ object_pose,
            ))
    if reverse_observations:
        observations.reverse()

    groups, actual = init_camera_groups(observations, [23, 99, 7, 12, 0])

    assert sorted(groups) == [[0, 7, 12, 23], [99]]
    for camera, transform in expected.items():
        np.testing.assert_allclose(actual[camera], transform, atol=1e-12)
    np.testing.assert_array_equal(actual[99], np.eye(4))
