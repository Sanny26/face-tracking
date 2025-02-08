import cv2
import os 
import json
import torch
import argparse
import numpy as np
from PIL import Image, ImageDraw 
from torch.nn.functional import cosine_similarity

from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceRecognition:
    def __init__(self, device):
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def crop_face(self, frame, box):
        sx, sy, ex, ey = map(int, box)
        cropped_image = frame.crop((sx, sy, ex, ey)).convert("RGB")
        return np.array(cropped_image)

    def get_embeddings(self, faces):
        faces = faces.to(self.device)
        return self.resnet(faces).detach().cpu()

    def detect_and_extract(self, frame):
        boxes, conf = self.mtcnn.detect(frame)
        faces = self.mtcnn.extract(frame, boxes, None) if boxes is not None else None
        return boxes, conf, faces

class VideoProcessor:
    def __init__(self, video_path, template_path, face_recognition,
                 detect_thresh=0.95, sim_thresh=0.5, merge_clips=0, draw_boxes=False):
        assert os.path.exists(template_path), f"Error: Template file '{self.template_path}' not found."
        assert os.path.exists(video_path), f"Error: Video file '{self.video_path}' not found."
        
        self.video_path = video_path
        self.template_path = template_path
        self.face_recognition = face_recognition
        self.frames = self.load_video()
        self.template_embedding = self.process_template()
        self.detect_thresh = detect_thresh
        self.sim_thresh = sim_thresh 
        self.draw_boxes = draw_boxes
        self.merge_clips = merge_clips
    
    def load_video(self):
        cap = cv2.VideoCapture(self.video_path)
        assert cap.isOpened(), f"Error: Unable to open video file '{self.video_path}'."
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert fps > 0, "Error: FPS could not be determined."
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            assert frame is not None, "Error: Retrieved an empty frame."
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        
        cap.release()
        assert frames, "Error: No frames were extracted."
        self.fps = fps  
        return frames

    def process_template(self):
        template = Image.open(self.template_path)
        assert template is not None, "Error: Failed to open the template image."
        
        if template.mode == "RGBA":
            template = template.convert("RGB")
        
        _, _, faces = self.face_recognition.detect_and_extract(template)
        assert faces is not None, "Error: No faces detected in the template."
        assert len(faces) == 1, "Error: More than 1 face found in the template reference!"
        
        return self.face_recognition.get_embeddings(faces)

    def process_frames(self):
        frames_tracked = []
        boxes_tracked = []
        for i, frame in enumerate(self.frames[1000:]):
            print(f'\rProcessing frame: {i + 1}', end='')
            frame_tracked, box_tracked = self.track_faces(frame)
            frames_tracked.append(frame_tracked.resize((640, 360), Image.BILINEAR))
            boxes_tracked.append(box_tracked)
            if self.draw_boxes:
                self.frames[i] = frame_tracked
        print('\nProcessing complete.')
        return frames_tracked, boxes_tracked

    def track_faces(self, frame):
        boxes, conf, faces = self.face_recognition.detect_and_extract(frame)
        if self.draw_boxes:
            frame_draw = frame.copy()
            draw = ImageDraw.Draw(frame_draw)

        valid_box = None 
        if boxes is not None and faces is not None:
            faces_em = self.face_recognition.get_embeddings(faces)
            # euclidean
            # faces_dist = torch.norm(self.template_embedding - faces_em, dim=1)
            # idx = torch.argmin(faces_dist)

            sim = cosine_similarity(self.template_embedding, faces_em)
            idx = torch.argmax(sim)
            if conf[idx] > self.detect_thresh and sim[idx] > self.sim_thresh:
                if self.draw_boxes:
                    draw.rectangle(boxes[idx].tolist(), outline=(255, 0, 0), width=6)    
                valid_box = boxes[idx].tolist()

        return frame_draw, valid_box

    def save_video(self, frames_tracked, output_path='video_tracked.mp4'):
        dim = frames_tracked[0].size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 25.0, dim)

        for frame in frames_tracked:
            video_writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        video_writer.release()
        print(f"Video saved at {output_path}")

    def merge_consecutive_clips(self, clips_info, max_gap):
        merged_clips = {}
        clip_keys = sorted(clips_info.keys(), key=lambda x: int(x.split("_")[1]))  # Sort by clip index
        merged_clip_count = 1

        prev_clip = None
        for clip_key in clip_keys:
            clip = clips_info[clip_key]
            if prev_clip is None:
                prev_clip = clip
                continue
            prev_end = prev_clip["clip_end"]
            curr_start = clip["clip_start"]
            if prev_end + max_gap >= curr_start:  
                prev_clip["clip_end"] = clip["clip_end"]
                prev_clip["end_time"] = clip["end_time"]
                # adding none for non-face frames while merging.
                gap_size = curr_start - prev_end - 1
                prev_clip["clip_boxes"].extend([None] * gap_size)
                prev_clip["clip_boxes"].extend(clip["clip_boxes"])
            else:
                merged_clips[f"clip_{merged_clip_count}"] = prev_clip
                merged_clip_count += 1
                prev_clip = clip

        if prev_clip is not None:
            merged_clips[f"clip_{merged_clip_count}"] = prev_clip
        return merged_clips


    def save_clips(self, boxes_tracked, save_dir='results'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        clips_info = {}
        clip_boxes = []
        clip_start = None
        clip_count = 1
        fps = self.fps
        frame_interval = 1 / fps

        for i, box in enumerate(boxes_tracked):
            if box is not None:
                if clip_start is None:
                    clip_start = i  
                x_min, y_min, x_max, y_max = map(int, box)
                width = x_max - x_min
                height = y_max - y_min
                clip_boxes.append([x_min, y_min, width, height])
            else:
                if clip_start is not None:
                    clip_end = i - 1
                    clips_info[f'clip_{clip_count}'] = {
                        "start_time": clip_start * frame_interval,
                        "end_time": clip_end * frame_interval,
                        "clip_start": clip_start,
                        "clip_end" : clip_end,
                        "clip_boxes": clip_boxes
                    }
                    clip_count += 1
                    clip_start = None
                    clip_boxes = []

        # Handle case where last frames form a clip
        if clip_start is not None:  
            clip_end = len(boxes_tracked) - 1
            clips_info[f'clip_{clip_count}'] = {
                "start_time": clip_start * frame_interval,
                "end_time": clip_end * frame_interval,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "clip_boxes": clip_boxes
            }

        if self.merge_clips > 0:
            clips_info = self.merge_consecutive_clips(clips_info, self.merge_clips)
        
        # Writing clips
        for clip_count, clip_data in clips_info.items():
            clip_start_frame = clip_data['clip_start']
            clip_end_frame = clip_data['clip_end']
            clip_path = os.path.join(save_dir, f'{clip_count}.mp4')
            self.save_video(self.frames[clip_start_frame:clip_end_frame+1],
                            output_path=clip_path)

            clip_data.pop('clip_start', None)
            clip_data.pop('clip_end', None)  
            clip_data['clip_fname'] = clip_path
            if clip_count == 3:
                break

        # Saving clip metadata
        json_path = os.path.join(save_dir, 'clips_metadata.json')
        with open(json_path, 'w') as f:
            json.dump(clips_info, f, indent=4)

        print(f"Saved clips in {save_dir} and metadata in {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Face Tacking")

    parser.add_argument('--video', type=str, required=True, help="Path to the input video file")
    parser.add_argument('--template', type=str, required=True, help="Path to the template image")
    parser.add_argument('--output', type=str, default='output_clips', help="Directory to save output clips")
    parser.add_argument('--detect-thresh', type=float, default=0.95, help="Detection threshold (default: 0.95)")
    parser.add_argument('--sim-thresh', type=float, default=0.5, help="Similarity threshold for face matching (default: 0.5)")
    parser.add_argument('--draw-boxes', action="store_true", help="Use this to draw bounding boxes in the generated clips")
    parser.add_argument('--merge-clips', type=int, default=0, help="Clips will be merged within 5 frames of no detection")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    face_recognition = FaceRecognition(device)
    video_processor = VideoProcessor(
        args.video, args.template, face_recognition, 
        detect_thresh=args.detect_thresh, sim_thresh=args.sim_thresh,
        merge_clips=args.merge_clips, draw_boxes=args.draw_boxes
    )

    tracked_frames, tracked_boxes = video_processor.process_frames()

    # Uncomment if you want to save the processed video
    # video_processor.save_video(tracked_frames, output_path='video_out2.mp4')

    video_processor.save_clips(tracked_boxes, save_dir=args.output)
