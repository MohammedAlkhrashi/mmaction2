# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
from multiprocessing import Pool
import mmcv
import subprocess
import time

def has_audio(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=nb_streams", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return (int(result.stdout) -1)


def extract_audio_wav(line):
    """Extract the audio wave from video streams using FFMPEG."""
    video_id, _ = osp.splitext(osp.basename(line))
    video_dir = osp.dirname(line)
    video_rel_dir = osp.relpath(video_dir, args.root)
    dst_dir = osp.join(args.dst_root, video_rel_dir)
    os.popen(f'mkdir -p {dst_dir}')
    audio_exists = has_audio(line)
    print(audio_exists)
    if not audio_exists:
        cmd = f"ffmpeg -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 -i {line} -c:v copy -c:a aac -shortest {line.replace('.avi','')}_noaudio.avi -y"
        line = line.replace('.avi','')+'_noaudio.avi'
        print('*'*1000, cmd)
        os.popen(cmd)
        time.sleep(1.5)
    try:
        if osp.exists(f'{dst_dir}/{video_id}.wav'):
            return
        cmd = f'ffmpeg -i {line}  -map 0:a  -y {dst_dir}/{video_id}.wav'
        os.popen(cmd)
    except BaseException:
        with open('extract_wav_err_file.txt', 'a+') as f:
            f.write(f'{line}\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Extract audios')
    parser.add_argument('root', type=str, help='source video directory')
    parser.add_argument('dst_root', type=str, help='output audio directory')
    parser.add_argument(
        '--level', type=int, default=2, help='directory level of data')
    parser.add_argument(
        '--ext',
        type=str,
        default='mp4',
        choices=['avi', 'mp4', 'webm'],
        help='video file extensions')
    parser.add_argument(
        '--num-workers', type=int, default=8, help='number of workers')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    mmcv.mkdir_or_exist(args.dst_root)

    print('Reading videos from folder: ', args.root)
    print('Extension of videos: ', args.ext)
    print(args.root + '/*' * args.level + '.' + args.ext)
    fullpath_list = glob.glob(args.root + '/*' * args.level + '.' + args.ext)
    done_fullpath_list = glob.glob(args.dst_root + '/*' * args.level + '.wav')
    print('Total number of videos found: ', len(fullpath_list))
    print('Total number of videos extracted finished: ',
          len(done_fullpath_list))
    print(fullpath_list)
    pool = Pool(args.num_workers)
    pool.map(extract_audio_wav, fullpath_list)
    cmd_clear = f"find . -type f -name '*noaudio*' -delete"
    os.popen(cmd_clear)
