import cv2
import torch
from facenet_pytorch import MTCNN
from imutils.video import FileVideoStream

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FastMTCNN(object):
    """Fast MTCNN implementation."""

    def __init__(self, stride=1, batch_size=60, image_size=256, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.

        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.

        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            select_largest=True
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.batch_size = batch_size

        self.mtcnn = MTCNN(image_size=image_size, select_largest=True, device=device, *args, **kwargs)

    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                for f in frames
            ]

        faces_subset = self.mtcnn(frames[::self.stride])

        faces = []
        for i, face in enumerate(faces_subset):
            faces += [face] * self.stride

        return faces


def detect_faces(fast_mtcnn, filename, max_frames=300):
    frames = []
    faces = []
    batch_size = 60
    v_cap = FileVideoStream(filename).start()
    v_len = min(max_frames, int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT)))
    # print(f"Total of {v_len} frames.")
    for j in range(v_len):
        frame = v_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) >= batch_size or j == v_len - 1:
            faces += fast_mtcnn(frames)
            frames = []
    v_cap.stop()
    return torch.stack(faces, dim=0)
