📹 Video-Unjumbler

Video-Unjumbler is a Python tool that attempts to reconstruct a jumbled (shuffled) sequence of video frames using visual similarity models such as OpenCLIP.
It extracts frames, embeds them, calculates similarity, and rebuilds a more logical sequence.

✅ Features

✅ Extract frames from video
✅ Compute frame similarity using OpenCLIP
✅ Reorder frames based on predicted continuity
✅ Save reconstructed video
✅ CLI support

📦 Requirements
Python

Python 3.10+ recommended.

Install dependencies
pip install -r requirements.txt


If clip is missing, install OpenCLIP:

pip install open_clip_torch


or original OpenAI CLIP:

pip install git+https://github.com/openai/CLIP.git

📁 Project Structure
video-unjumbler/
│
├─ src/
│   ├─ Unjumbler.py         # Main script
│   ├─ utils.py             # Helpers
│   └─ ...
│
├─ output/
│   ├─ frames/              # Extracted frames
│   └─ result.mp4           # Reordered video
│
├─ requirements.txt
└─ README.md

▶️ Run
python src/Unjumbler.py --input jumbled.mp4 --output result.mp4

Optional flags
Flag	Description
--input	Input video file
--output	Output restored video
--save-frames	Save extracted frames
--no-clip	Run without CLIP

Example:

python src/Unjumbler.py --input jumbled.mp4 --output restored.mp4 --save-frames

⚙️ How it Works

Extract video frames using OpenCV

Generate embeddings using OpenCLIP

Compute similarity between frames

Determine best ordering

Rebuild video from reordered frames

📚 Installation Notes

To avoid ModuleNotFoundError: No module named 'clip', install OpenCLIP:

pip install open_clip_torch


Or install OpenAI CLIP:

pip install git+https://github.com/openai/CLIP.git


To be safe, install into the same Python interpreter used to run the file:

"<path to python.exe>" -m pip install open_clip_torch

❗ Troubleshooting
❌ ModuleNotFoundError: No module named 'clip'

✅ Install OpenCLIP:

pip install open_clip_torch


✅ Or install OpenAI CLIP:

pip install git+https://github.com/openai/CLIP.git

❌ Wrong environment

Check python being used:

where python
where pip


Use:

python -m pip install open_clip_torch

✅ Future Improvements

🔹 Optical Flow support
🔹 Audio-guided ordering
🔹 Motion-aware ordering
🔹 Automatic grouping of scenes
