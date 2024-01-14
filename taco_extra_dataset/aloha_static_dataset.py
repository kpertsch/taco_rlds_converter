from typing import Iterator, Tuple, Any

import copy
import cv2
import glob
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from aloha_static_dataset.conversion_utils import MultiThreadedDatasetBuilder


FILE_PATH = '/nfs/kun2/datasets/taco/taco_extra_processed_15hz_resize'


def _generate_examples(ids_and_annotations) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""

    def _parse_example(ids, annotation):
        episode = []
        for i, id in enumerate(range(ids[0], ids[1]+1)):
            try:
                data = np.load(os.path.join(FILE_PATH, f'episode_{id}.npz'))
            except:
                print(f"Failed to load {os.path.join(FILE_PATH, f'episode_{id}.npz')}")
                return None

            episode.append({
                'observation': {
                    'rgb_static': data['rgb_static'],
                    'rgb_gripper': data['rgb_gripper'],
                    'robot_obs': data['robot_obs'],
                    'natural_language_instruction': annotation,
                },
                'action': {
                    'actions': data['actions'],
                },
                'discount': 1.0,
                'reward': float(i == (ids[1]-ids[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (ids[1]-ids[0] - 1),
                'is_terminal': i == (ids[1]-ids[0] - 1),
            })

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return str(ids), sample

    # for smallish datasets, use single-thread parsing
    for ids, annotation in ids_and_annotations:
        yield _parse_example(ids, annotation)


class TacoExtra(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 40              # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 80   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'rgb_static': tfds.features.Image(
                            shape=(150, 200, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='RGB camera observation.',
                        ),
                        'rgb_gripper': tfds.features.Image(
                            shape=(150, 200, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='RGB gripper camera observation.',
                        ),
                        'robot_obs': tfds.features.Tensor(
                            shape=(15,),
                            dtype=np.float32,
                            doc='EE position (3), EE orientation in euler angles (3), '
                                'gripper width (1), joint positions (7), gripper action (1).',
                        ),
                        'natural_language_instruction': tfds.features.Text(
                            doc='Language Instruction.'
                        ),
                    }),
                    'action': tfds.features.FeaturesDict({
                        'actions': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='absolute desired values for gripper pose '
                                '(first 6 dimensions are x, y, z, yaw, pitch, roll), '
                                'last dimension is open_gripper (-1 is open gripper, 1 is close).',
                        ),
                    }),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),

                }),
                'episode_metadata': tfds.features.FeaturesDict({
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        annotations = np.load(os.path.join(FILE_PATH, "annotations/auto_lang_ann.npy"), allow_pickle=True)
        frame_ids = annotations.item()['info']['indx']
        language_instructions = annotations.item()['language']['ann']

        return {
            'train': [*zip(frame_ids, language_instructions)],
        }

