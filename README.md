# 📹 Video-Unjumbler

Video-Unjumbler is a Python tool that attempts to **reconstruct a jumbled (shuffled) sequence of video frames** using visual similarity models such as **OpenCLIP**.  
It extracts frames, embeds them, calculates similarity, and rebuilds a more logical sequence.

---

## ✅ Features

- ✅ Extract frames from video  
- ✅ Compute frame similarity using OpenCLIP  
- ✅ Reorder frames based on predicted continuity  
- ✅ Save reconstructed video  
- ✅ CLI support  

---

## 📦 Requirements

### Python
Python **3.10+** recommended.

### Install dependencies
```bash
pip install -r requirements.txt
If clip is missing, install OpenCLIP:
```
```bash
pip install open_clip_torch
```
Or install original OpenAI CLIP:
```bash
pip install git+https://github.com/openai/CLIP.git
```
📁 Project Structure
bash
Copy code
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
```bash
python src/Unjumbler.py --input jumbled.mp4 --output result.mp4
```
Optional flags
Flag	Description
--input	Input video file
--output	Output restored video
--save-frames	Save extracted frames
--no-clip	Run without CLIP

Example:

```bash
python src/Unjumbler.py --input jumbled.mp4 --output restored.mp4 --save-frames
```
⚙️ How It Works
Extract video frames using OpenCV

Generate embeddings using OpenCLIP

Compute similarity between frames

Determine best ordering

Rebuild video from reordered frames

📚 Installation Notes
To avoid error:

```vbnet
ModuleNotFoundError: No module named 'clip'
```
Install OpenCLIP:
```bash
pip install open_clip_torch

```
Or install OpenAI CLIP:

```bash
pip install git+https://github.com/openai/CLIP.git
```
To be safe, install into the same Python interpreter running your script:

```bash
"<path_to_python.exe>" -m pip install open_clip_torch
```
❗ Troubleshooting
❌ ModuleNotFoundError: No module named 'clip'
✅ Install OpenCLIP:

```bash
pip install open_clip_torch
```
✅ Or install OpenAI CLIP:
```bash
pip install git+https://github.com/openai/CLIP.git
```
❌ Wrong environment
Check Python being used:

```bash
where python
where pip
```
Install properly:
```bash
python -m pip install open_clip_torch
```
