from __future__ import division
import cv2
from pathlib import Path
from random import seed
from random import randint

class VideoSequenceGenerator:
    def __init__(self, clip_size):
        self.clip_size = clip_size

    def _create_output_dir(self, path):
        """Create and return the output directory path."""
        parent = Path(path).parent.parent
        out_dir = parent / 'sequences' / f'full_game_goal_cuts_seq_{self.clip_size}'
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    @staticmethod
    def _get_video_info(video, path):
        """Extract FPS, frame count, and filename from video."""
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        filename = Path(path).name
        return fps, frame_count, filename

    @staticmethod
    def _shift_video(video, frame_num):
        """Seek video to specific frame."""
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    def _already_exists(self, output_path):
        """Check if output file exists and has correct frame count."""
        if not output_path.exists():
            return False
        video = cv2.VideoCapture(str(output_path))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return frame_count == self.clip_size

    def _write_sequence(self, output_path, frames, fps, width, height):
        """Write frames to video file."""
        if output_path.exists() and self._already_exists(output_path):
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()

    def _compute_labels(self, frame_idx, clip_size, pad_frames, vid_len, ignore_frame):
        """Compute highlight and celebration labels for a sequence."""
        start_idx = frame_idx - clip_size + 1
        is_highlight = not (start_idx < pad_frames or start_idx >= (vid_len - pad_frames))
        is_celebration = (frame_idx - clip_size) >= ignore_frame
        return is_highlight, is_celebration, start_idx

    def generate(self, vid_path, max_frame, pad_frames, ignore_frame, generate=True):
        """
        Generate sequential video clips from a source video.
        
        Args:
            vid_path: Path to source video
            max_frame: Reference frame for goal/highlight detection
            pad_frames: Number of frames to skip at start/end
            ignore_frame: Frame index after which to mark as celebration
            generate: Whether to generate clips (if False, only scans video)
        
        Returns:
            Tuple of (info_list, output_dir) or (None, output_dir) if no clips generated
        """
        video = cv2.VideoCapture(vid_path)
        vid_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        out_dir = self._create_output_dir(vid_path)
        info_list = []

        if not generate:
            video.release()
            return None, out_dir

        fps, _, filename = self._get_video_info(video, vid_path)
        highlight_len = vid_len - 2 * pad_frames
        frames = []
        frame_idx = pad_frames
        seq_num = 0

        self._shift_video(video, pad_frames)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            frames.append(frame)

            # Check if we have a complete clip
            if (frame_idx != 0 and len(frames) == self.clip_size and 
                frame_idx <= vid_len - pad_frames):
                seq_num += 1
                output_path = out_dir / f'seq_{seq_num}__{filename}'

                self._write_sequence(output_path, frames, fps, width, height)
                
                is_highlight, is_celebration, start_idx = self._compute_labels(
                    frame_idx, self.clip_size, pad_frames, vid_len, ignore_frame)

                info_list.append([
                    str(output_path), is_highlight, is_celebration, start_idx,
                    max_frame, highlight_len, pad_frames
                ])
                frames = []

            frame_idx += 1

        video.release()
        return (info_list, out_dir) if info_list else (None, out_dir)

    def generate_with_random_offsets(self, vid_path, max_frame, pad_frames, ignore_frame, 
                                     generate=True, num_samples=None):
        """
        Generate clips at random frame positions.
        
        Args:
            vid_path: Path to source video
            max_frame: Reference frame for goal detection
            pad_frames: Number of padding frames
            ignore_frame: Celebration start frame
            generate: Whether to generate clips
            num_samples: Number of random samples (default: 4 * estimated sequences)
        """
        video = cv2.VideoCapture(vid_path)
        fps, vid_len, filename = self._get_video_info(video, vid_path)
        out_dir = self._create_output_dir(vid_path)
        info_list = []

        if not generate:
            video.release()
            return None, out_dir

        no_highlight_seqs = 2
        useless_frames = pad_frames - (no_highlight_seqs * self.clip_size)
        highlight_len = vid_len - 2 * pad_frames
        
        if num_samples is None:
            tot_sequences = no_highlight_seqs + int(highlight_len / self.clip_size)
            num_samples = 4 * tot_sequences

        for _ in range(num_samples):
            shift = randint(useless_frames, int(pad_frames + highlight_len - self.clip_size))
            output_path = out_dir / f'seq_frame{shift}__{filename}'
            
            if self._already_exists(output_path):
                continue

            self._shift_video(video, shift)
            frames = []
            
            while len(frames) < self.clip_size:
                ret, frame = video.read()
                if not ret:
                    break
                height, width = frame.shape[:2]
                frames.append(frame)

            if len(frames) == self.clip_size:
                self._write_sequence(output_path, frames, fps, width, height)
                
                is_highlight = shift > pad_frames - self.clip_size / 5
                is_celebration = shift >= ignore_frame - self.clip_size / 2
                
                info_list.append([
                    str(output_path), is_highlight, is_celebration, shift - useless_frames,
                    max_frame, highlight_len, no_highlight_seqs * self.clip_size, ignore_frame
                ])

        video.release()
        return (info_list, out_dir) if info_list else (None, out_dir)

    def generate_with_fixed_offsets(self, vid_path, max_frame, pad_frames, ignore_frame, 
                                    generate=True, step=10):
        """
        Generate clips at fixed intervals across the video.
        
        Args:
            vid_path: Path to source video
            max_frame: Reference frame for goal detection
            pad_frames: Number of padding frames
            ignore_frame: Celebration start frame
            generate: Whether to generate clips
            step: Frame interval between clips (default: 10)
        """
        video = cv2.VideoCapture(vid_path)
        fps, vid_len, filename = self._get_video_info(video, vid_path)
        out_dir = self._create_output_dir(vid_path)
        info_list = []

        if not generate:
            video.release()
            return None, out_dir

        no_highlight_seqs = 2
        useless_frames = pad_frames - (no_highlight_seqs * self.clip_size)
        highlight_len = vid_len - 2 * pad_frames
        
        for shift in range(useless_frames, int(pad_frames + highlight_len - self.clip_size + 1), step):
            output_path = out_dir / f'seq_frame{shift}__{filename}'
            
            if self._already_exists(output_path):
                continue

            self._shift_video(video, shift)
            frames = []
            
            while len(frames) < self.clip_size:
                ret, frame = video.read()
                if not ret:
                    break
                height, width = frame.shape[:2]
                frames.append(frame)

            if len(frames) == self.clip_size:
                self._write_sequence(output_path, frames, fps, width, height)
                
                is_highlight = shift > pad_frames - self.clip_size / 5
                is_celebration = shift >= ignore_frame - self.clip_size / 2
                
                info_list.append([
                    str(output_path), is_highlight, is_celebration, shift - useless_frames,
                    max_frame, highlight_len, no_highlight_seqs * self.clip_size, ignore_frame
                ])

        video.release()
        return (info_list, out_dir) if info_list else (None, out_dir)