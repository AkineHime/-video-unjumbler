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
Installation:
```
git clone https://github.com/AkineHime/-video-unjumbler.git
```
Then enter the project:
```
bash
cd -video-unjumbler
```

## 📦 Requirements

### Python
Python **3.10+** recommended.

### Install dependencies
```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```
If clip is missing, install CLIP:
```bash
pip install git+https://github.com/openai/CLIP.git
```
📁 Project Structure
```bash
video-unjumbler/
│
├─ src/
│   ├─ Unjumbler.py         # CLI
│   ├─ Unjumbler_gui.py     # GUI version 
│   └─ requirements.txt
│
└─ README.md
```
CLI:
Put the video to be Unjumnled in the src folder
and name it jumbled.mp4(anthing you want) 

GUI:
for gui install PySide6
```
bash
python -m pip install PySide6
```
if you are in a virtual enviornment
```bash
.\.venv\Scripts\python -m pip install PySide6
```
Open the python file and select the input file.
give the output file name.
you can preview the input/output file.

▶️ Run CLI(in -video-unjumbler folder)
```bash
 python src/Unjumbler.py --input input/<File name>.mp4 --output output/<File name>.mp4
```
Example:
```bash
 python src/Unjumbler.py --input input/jumbled_video.mp4 --output output/result.mp4
```
▶️ Run GUI
```bash
python src/Unjumbler_gui.py --gui --input input/<File name>.mp4 --output output/<File name>.mp4
```
Example:
```bash
python src/Unjumbler_gui.py --gui --input input/jumbled_video.mp4 --output output/result.mp4
```
Optional flags
Flag	Description
--input	Input video file
--output	Output restored video
--save-frames	Save extracted frames
--no-clip	Run without CLIP
--reverse to reverse the output(no argument)
--fps FPS to define fps(auto detected)

Example:
CLI:
```bash
python src/Unjumbler.py --input <File name>.mp4 --output <File name>.mp4 --save-frames --reverse --fps 30 --no-clip
```
GUI:
```bash
python src/Unjumbler_gui.py --gui --input <File name>.mp4 --output <File name>.mp4 
```
Not Applicable for GUI:
--save-frames
--reverse --fps 30
--no-clip

⚙️ How It Works
=>Extract video frames using OpenCV

=>Generate embeddings using OpenCLIP

=>Compute similarity between frames

=>Determine best ordering

=>Rebuild video from reordered frames

📚 Installation Notes
To avoid error:

❌ Wrong environment
Check Python being used:

```bash
where python
where pip
```
