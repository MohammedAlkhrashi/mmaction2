# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

from .builder import DATASETS
from .rawframe_dataset import RawframeDataset


@DATASETS.register_module()
class AudioVisualDatasetMk(RawframeDataset):
    def __init__(self, ann_file, pipeline, audio_prefix, **kwargs):
        self.audio_prefix = audio_prefix
        self.video_prefix = kwargs.pop('video_prefix', None)
        self.data_prefix = kwargs.get('data_prefix', None)
        super().__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self):
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split() 
                video_info = {}
                idx = 0
                # idx for frame_dir
                frame_dir = line_split[idx]
                frame_dir = osp.splitext(frame_dir)[0] # remove file    extension
                if self.audio_prefix is not None:
                    audio_path = osp.join(self.audio_prefix,
                                          frame_dir + '.wav')
                    if not osp.isfile(audio_path):
                        print("File not fond: ", audio_path)
                        continue
                    video_info['audio_path'] = audio_path
                if self.video_prefix:
                    video_path = osp.join(self.video_prefix,
                                          frame_dir + '.avi')
                    video_info['filename'] = video_path

                # if self.data_prefix is not None:
                #     frame_dir = osp.join(self.data_prefix, frame_dir)
                #     video_info['frame_dir'] = frame_dir
                idx += 1
                if self.with_offset:
                    # idx for offset and total_frames
                    video_info['offset'] = int(line_split[idx])
                    video_info['total_frames'] = int(line_split[idx + 1])
                    idx += 2
                else:
                    # idx for total_frames
                    video_info['total_frames'] = int(line_split[idx])
                    idx += 1
                # idx for label[s]
                label = [int(x) for x in line_split[idx:]]
                assert len(label) != 0, f'missing label in line: {line}'
                if self.multi_class:
                    assert self.num_classes is not None
                    video_info['label'] = label
                else:
                    assert len(label) == 1
                    video_info['label'] = label[0]
                video_infos.append(video_info)
        print(f"len video_infos = {len(video_infos)}")
        return video_infos
