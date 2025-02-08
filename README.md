## Simple Face Tracking

- A lightweight face detection pipeline followed by embedding comparison to track faces across a video.
- Uses FaceNet for face detection.
- Employs Inception ResNet for comparing detected faces with template images.

### Sample Results
The below shown results are generated for this [YouTube video](https://www.youtube.com/watch?v=cmkZeTX5fq0).

<p align="center">
  <img src="samples/3.gif" alt="Face Tracking Results" width="720"/><br/>
  <em>Tracked face shown in red</em>
</p>

In the below results, consecutive clips are merged if tracking is lost for 2 frames or fewer.
<p align="center">
  <img src="samples/1.gif" alt="Face Tracking Results" width="720"/><br/>
  <em>Tracked face shown in red, merged clips within 2 clips of no tracking</em>
</p>

### Setup and Instructions

#### Dependencies and Env setup
Follow the below to set up the env for the repo.
```bash
conda create --name facenet_env python=3.10
conda activate facenet_env

pip install facenet-pytorch opencv-python
```

#### Run the script

To run the script, please use the following:

```bash 
python main.py --video data/downloaded_video.mp4 \
--template data/template3.jpg \
--output results \
--detect-thresh 0.95 \
--sim-thresh 0.5 \
--draw-boxes
```

Use with ```--merge-clips <n_frame>``` to merge consecutive clips even if tracking is lost for <n_frames> frames or fewer. In this case, the boxes in the metadata is saved as None. We could to interpolate the boxes within these frames to improve results.


## Assumptions Limitations
- Clean, frontal face image with high resolution (e.g., 512x512 pixels or higher).
- Ideal for minimizing noise and ensuring accurate embeddings.
- Real-time performance is not a requirement; model accuracy is prioritized.
- Requires significant memory for holding pre-trained models like VGG and ResNet (can be 1-2GB or more, depending on the model).
- Video frames are processed sequentially; videos can be loaded entirely into memory for efficient processing.
- Frames are cached until processed by the face detection module, leading to higher memory usage.
- Emphasis on extracting all possible clips; the system may be memory-intensive when handling long videos with many faces.
- Not ideal for detecting upside-down or highly rotated faces due to limitations in the face detector's orientation tolerance.

## Potential Improvements
- **Multi-scale reference embeddings**: Enhance tracking by creating multiple embeddings from the template.
- **Speed optimization**: Skip frames and reuse previously detected face boxes with margin (optical flow).
- **Motion Blur**: Better handling of motion blur & dynamic movement (e.g., dancing).
- **Scalability:** Process large videos efficiently by loading them in chunks.
- **Adaptive tracking:** Build an embedding tree as the video progresses to handle extreme face poses and varying lighting conditions.
- **Improve racial bias** Evaluate the model for any racial bias and finetune the face detection models accordingly

## Sources

- [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch)
- Data Sources and Copyright: Google & YouTube

