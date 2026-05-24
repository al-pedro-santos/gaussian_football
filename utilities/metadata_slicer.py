from __future__ import division
import cv2
import numpy as np
from pathlib import Path
import os
from random import randint
from decord import VideoReader, cpu


class VideoSequenceGenerator:
    def __init__(self, clip_size):
        self.clip_size = clip_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self, vid_path):
        """Return a VideoReader and basic metadata."""
        vr = VideoReader(vid_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        vid_len = len(vr)
        video_name_with_fmt = vid_path.split(os.path.sep)[-1]
        return vr, fps, vid_len, video_name_with_fmt

    def create_path(self, path):
        parent = Path(path).parent.parent
        out_dir = parent / 'sequences'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dir = out_dir / ('full_game_goal_cuts_seq_' + str(self.clip_size))
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def alredy_exists(self, out_vid_path):
        """Return True if the clip file already exists with the correct frame count."""
        if os.path.isfile(out_vid_path):
            check = cv2.VideoCapture(str(out_vid_path))
            count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
            check.release()
            return count == self.clip_size
        return False

    def write_sequence(self, out_vid_path, frames, fps, width, height):
        """
        Write a list of numpy frames (RGB, from decord) to disk as an MP4.
        Converts RGB -> BGR for cv2.VideoWriter.
        Skips writing if a valid clip already exists.
        """
        if self.alredy_exists(out_vid_path):
            return
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (width, height))
        for frame in frames:
            out.write(cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR))
        out.release()

    def get_seq_state(self, num_seq, no_high_seqs, i, ignore_frame):
        is_highlight = num_seq > no_high_seqs
        is_celebration = (i - self.clip_size) >= ignore_frame
        start_seq = i - self.clip_size
        return is_highlight, is_celebration, start_seq

    def get_seq_state_r(self, length, shift, pad_frames, ignore_frame):
        is_highlight = shift > pad_frames - length / 5
        is_celebration = shift >= ignore_frame - length / 2
        return is_highlight, is_celebration

    # ------------------------------------------------------------------
    # generate  –  original sliding-window approach
    # ------------------------------------------------------------------

    def generate(self, vid_path, max_frame, pad_frames, ignore_frame, generate):
        print("*")
        info_list = []
        out_dir = self.create_path(vid_path)

        vr, fps, vid_len, video_name_with_fmt = self._open(vid_path)
        parent_highlight_len = vid_len - 2 * pad_frames

        if generate:
            frames = []
            num_seq = 0
            i = pad_frames  # mirrors original loop counter

            for frame_idx in range(pad_frames, vid_len):
                frame = vr[frame_idx]          # decord: single-frame seek
                h, w = frame.shape[:2]
                frames.append(frame)

                if len(frames) == self.clip_size and i <= vid_len - pad_frames:
                    num_seq += 1
                    out_vid_path = out_dir / ('seq_' + str(num_seq) + '__' + video_name_with_fmt)
                    self.write_sequence(out_vid_path, frames, fps, w, h)

                    is_highlight = (
                        pad_frames <= (i - self.clip_size + 1) < (vid_len - pad_frames)
                    )
                    is_celebration = (i - self.clip_size) >= ignore_frame
                    start_seq = i - self.clip_size + 1

                    info_list.append([
                        str(out_vid_path), is_highlight, is_celebration,
                        start_seq, max_frame, parent_highlight_len, pad_frames
                    ])
                    frames = []

                i += 1

        if info_list:
            return info_list, out_dir
        return None, out_dir

    # ------------------------------------------------------------------
    # generate2  –  random-sampling approach
    # ------------------------------------------------------------------

    def generate2(self, vid_path, max_frame, pad_frames, ignore_frame, generate):
        print("*")
        out_dir = self.create_path(vid_path)
        info_list = []

        no_high_seqs = 2
        no_high_seq_len = no_high_seqs * self.clip_size
        useless_frames = pad_frames - no_high_seq_len
        max_frame -= useless_frames
        ignore_frame -= useless_frames

        if generate:
            vr, fps, vid_len, video_name_with_fmt = self._open(vid_path)
            real_highlight_len = vid_len - 2 * pad_frames
            tot_sequences = no_high_seqs + int(real_highlight_len / self.clip_size)

            sample_frame = vr[0]
            h, w = sample_frame.shape[:2]

            for _ in range(4 * tot_sequences):
                shift = randint(
                    useless_frames,
                    pad_frames + real_highlight_len - self.clip_size
                )
                out_vid_path = out_dir / ('seq_frame' + str(shift) + '__' + video_name_with_fmt)

                # decord batch fetch: indices for this clip
                indices = list(range(shift, shift + self.clip_size))
                frames = vr.get_batch(indices)   # NDArray shape (T, H, W, C)

                if not self.alredy_exists(out_vid_path):
                    self.write_sequence(out_vid_path, [frames[t] for t in range(self.clip_size)], fps, w, h)

                is_highlight, is_celebration = self.get_seq_state_r(
                    self.clip_size, shift, pad_frames, ignore_frame
                )
                print(str(out_vid_path))
                info_list.append([
                    str(out_vid_path), is_highlight, is_celebration,
                    shift - useless_frames, max_frame,
                    real_highlight_len, no_high_seq_len, ignore_frame
                ])

        if info_list:
            return info_list, out_dir
        return None, out_dir

    # ------------------------------------------------------------------
    # generate_all  –  dense stride-10 sampling
    # ------------------------------------------------------------------

    def generate_all(self, vid_path, max_frame, pad_frames, ignore_frame, generate):
        print("*")
        out_dir = self.create_path(vid_path)
        info_list = []

        no_high_seqs = 2
        no_high_seq_len = no_high_seqs * self.clip_size
        useless_frames = pad_frames - no_high_seq_len
        max_frame -= useless_frames
        ignore_frame -= useless_frames

        if generate:
            vr, fps, vid_len, video_name_with_fmt = self._open(vid_path)
            real_highlight_len = vid_len - 2 * pad_frames

            sample_frame = vr[0]
            h, w = sample_frame.shape[:2]

            start = useless_frames
            end = pad_frames + int(real_highlight_len) - self.clip_size + 1

            for shift in range(start, end, 10):
                print(shift)
                out_vid_path = out_dir / ('seq_frame' + str(shift) + '__' + video_name_with_fmt)

                indices = list(range(shift, shift + self.clip_size))
                frames = vr.get_batch(indices)

                if not self.alredy_exists(out_vid_path):
                    self.write_sequence(out_vid_path, [frames[t] for t in range(self.clip_size)], fps, w, h)

                is_highlight, is_celebration = self.get_seq_state_r(
                    self.clip_size, shift, pad_frames, ignore_frame
                )
                info_list.append([
                    str(out_vid_path), is_highlight, is_celebration,
                    shift - useless_frames, max_frame,
                    real_highlight_len, no_high_seq_len, ignore_frame
                ])

        if info_list:
            return info_list, out_dir
        return None, out_dir

    # ------------------------------------------------------------------
    # generate2_old  –  kept for reference, also ported
    # ------------------------------------------------------------------

    def generate2_old(self, vid_path, max_frame, pad_frames, ignore_frame, generate):
        print("*")
        out_dir = self.create_path(vid_path)
        info_list = []
        no_high_seqs = 2

        if generate:
            vr, fps, vid_len, video_name_with_fmt = self._open(vid_path)
            full_highlight_len = vid_len - 2 * pad_frames
            start_frame = pad_frames - no_high_seqs * self.clip_size
            ignore_frame = ignore_frame - start_frame
            tot_sequences = no_high_seqs + int(full_highlight_len / self.clip_size)

            sample_frame = vr[0]
            h, w = sample_frame.shape[:2]

            num_seq = 0
            i = 1
            frame_idx = start_frame

            while num_seq < tot_sequences:
                indices = list(range(frame_idx, frame_idx + self.clip_size))
                if indices[-1] >= vid_len:
                    break
                frames = vr.get_batch(indices)

                num_seq += 1
                out_vid_path = out_dir / ('seq_' + str(num_seq) + '__' + video_name_with_fmt)
                self.write_sequence(out_vid_path, [frames[t] for t in range(self.clip_size)], fps, w, h)

                is_highlight, is_celebration, start_seq = self.get_seq_state(
                    num_seq, no_high_seqs, i, ignore_frame
                )
                info_list.append([
                    str(out_vid_path), is_highlight, is_celebration,
                    start_seq, max_frame, full_highlight_len, pad_frames
                ])

                frame_idx += self.clip_size
                i += self.clip_size

        if info_list:
            return info_list, out_dir
        return None, out_dir