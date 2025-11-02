import os
import sys
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import clip
from datetime import datetime


# =========================================================
# PRETTY LOGGING
# =========================================================
def pretty(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")

    icons = {
        "INFO": "ℹ️",
        "STEP": "🔄",
        "OK": "✅",
        "WARN": "⚠️",
        "ERR": "❌"
    }
    icon = icons.get(level, "")
    return f"[{ts}]  {icon}  {msg}"


# =========================================================
# FRAME EXTRACTION / VIDEO SAVING
# =========================================================
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    ok = True
    while ok:
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames read. Check --input path / codec.")

    return frames


def save_video(frames, output_path, fps=30):
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
    if not writer.isOpened():
        for fourcc in ("XVID", "MJPG", "avc1"):
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if writer.isOpened():
                break

    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open. Try other extension/codecs.")

    for f in frames:
        writer.write(f)
    writer.release()


# =========================================================
# FRAME PREPROCESS + FEATURES
# =========================================================
def downscale_frame(f, max_side=640):
    h, w = f.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return f
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)


def embed_frames(frames, device, max_side=640, model_name="ViT-B/32"):
    model, preprocess = clip.load(model_name, device=device)
    embs = []
    for f in tqdm(frames, desc="Embedding (CLIP)"):
        fs = downscale_frame(f, max_side=max_side)
        img = cv2.cvtColor(fs, cv2.COLOR_BGR2RGB)
        tens = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            e = model.encode_image(tens).float()
            e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
            e = e.cpu().numpy()[0]
        embs.append(e)
    return np.vstack(embs)


def hsv_hist(frames, bins=(32, 32, 8), max_side=480):
    hists = []
    for f in tqdm(frames, desc="Computing HSV hist"):
        fs = downscale_frame(f, max_side=max_side)
        hsv = cv2.cvtColor(fs, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                            [0, 180, 0, 256, 0, 256]).astype(np.float32)
        hist = hist.flatten()
        hist /= (np.linalg.norm(hist) + 1e-8)
        hists.append(hist)
    return np.vstack(hists)


# =========================================================
# SIMILARITY
# =========================================================
def cosine_sim(A):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    return A @ A.T


def mutual_knn_bonus(S, k=10, bonus=0.15):
    N = S.shape[0]
    idx = np.argsort(-S, axis=1)[:, :k]
    mask = np.zeros_like(S, dtype=np.float32)

    for i in range(N):
        nbrs = idx[i]
        for j in nbrs:
            if i in idx[j]:
                mask[i][j] = bonus

    np.fill_diagonal(mask, 0.0)
    return mask


def combined_similarity(clip_emb, hsv_emb, w_clip=0.70, w_hsv=0.30,
                        mknn_k=10, mknn_bonus=0.15):
    S_clip = cosine_sim(clip_emb).astype(np.float32)
    S_hsv = cosine_sim(hsv_emb).astype(np.float32)
    S = w_clip * S_clip + w_hsv * S_hsv
    S += mutual_knn_bonus(S, k=mknn_k, bonus=mknn_bonus)
    np.fill_diagonal(S, -1.0)
    return S


# =========================================================
# OPTICAL FLOW
# =========================================================
def warp_with_flow(img, flow):
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    mx = (gx + flow[..., 0]).astype(np.float32)
    my = (gy + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def flow_residual(a, b):
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    try:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(a_gray, b_gray, None)
    except:
        flow = cv2.calcOpticalFlowFarneback(a_gray, b_gray, None, 0.5,
                                            3, 25, 3, 5, 1.2, 0)
    a_warp = warp_with_flow(a_gray, flow)
    diff = (a_warp.astype(np.float32) - b_gray.astype(np.float32))
    return float(np.mean(np.abs(diff)) / 255.0)


def add_flow_to_similarity(frames, S, topk=40, side=384, w_flow=0.35):
    N = len(frames)
    small = [downscale_frame(f, max_side=side) for f in frames]
    S2 = S.copy()
    idx = np.argsort(-S, axis=1)[:, :topk]

    for i in tqdm(range(N), desc="Flow refine (top-k)"):
        for j in idx[i]:
            if i == j:
                continue
            a, b = small[i], small[j]
            res = flow_residual(a, b)
            s = np.exp(-5.0 * res)
            S2[i][j] = (1 - w_flow) * S2[i][j] + w_flow * s

    return S2


# =========================================================
# ORDER SEARCH HELPERS
# =========================================================
def spectral_seriation(S):
    D = 1.0 - np.maximum(S, -1.0)
    D -= D.min()
    D = (D + D.T) * 0.5
    W = np.exp(-D / (np.mean(D) + 1e-8))
    np.fill_diagonal(W, 0.0)
    d = np.sum(W, axis=1)
    L = np.diag(d) - W
    vals, vecs = np.linalg.eigh(L)

    if len(vals) < 2:
        return list(range(S.shape[0]))

    fiedler = vecs[:, 1]
    return np.argsort(fiedler).tolist()


def path_score(order, S):
    return float(sum(S[order[i], order[i + 1]] for i in range(len(order) - 1)))


def beam_search(S, start_order=None, beam_width=14, look_k=32):
    N = S.shape[0]

    if start_order is None:
        start = int(np.argmax(S.sum(axis=1)))
        beams = [([start], {start})]
    else:
        beams = [(start_order[:1], {start_order[0]})]

    for _ in range(1, N):
        new_beams = []
        for path, used in beams:
            last = path[-1]
            cand = np.argsort(-S[last])[:look_k]
            cnt = 0

            for c in cand:
                if c in used:
                    continue
                newp = path + [c]
                newu = set(used)
                newu.add(c)
                sc = path_score(newp, S)
                new_beams.append((newp, newu, sc))
                cnt += 1
                if cnt >= beam_width * 2:
                    break

        if not new_beams:
            break

        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = [(p, u) for (p, u, _) in new_beams[:beam_width]]

    best = max(beams, key=lambda bu: (len(bu[0]), path_score(bu[0], S)))[0]

    remaining = list(set(range(N)) - set(best))

    cur = best[-1]
    while remaining:
        nxt = remaining[int(np.argmax(S[cur, remaining]))]
        best.append(nxt)
        remaining.remove(nxt)
        cur = nxt

    return best


def two_opt(order, S, iters=300):
    n = len(order)
    best = order[:]

    for _ in range(iters):
        improved = False
        for i in range(1, n - 2):
            a, b = best[i - 1], best[i]
            for j in range(i + 1, n - 1):
                c, d = best[j], best[j + 1]
                before = S[a][b] + S[c][d]
                after = S[a][c] + S[b][d]
                if after > before:
                    best[i:j + 1] = reversed(best[i:j + 1])
                    improved = True
        if not improved:
            break
    return best


def local_repair(order, S, window=6, rounds=2):
    n = len(order)
    arr = order[:]

    for _ in range(rounds):
        changed = False
        for start in range(0, n - window + 1):
            seg = arr[start:start + window]
            base = path_score(seg, S)
            best_seg = seg[:]
            best_sc = base

            for i in range(len(seg)):
                for j in range(i + 1, len(seg)):
                    c2 = seg[:]
                    c2[i], c2[j] = c2[j], c2[i]
                    sc = path_score(c2, S)
                    if sc > best_sc:
                        best_sc = sc
                        best_seg = c2[:]

            if best_seg != seg:
                arr[start:start + window] = best_seg
                changed = True

        if not changed:
            break

    return arr


# =========================================================
# MAIN UNJUMBLE WRAPPER
# =========================================================
def unjumble_then_optionally_reverse(
        input_path,
        output_path,
        reverse_after=False,
        fps=30,
        progress_cb=None,
        progress_step_cb=None
):

    def log(msg, level="INFO", pct=None):
        txt = pretty(msg, level)
        if progress_cb:
            progress_cb(txt)
        else:
            print(txt)
        if pct is not None and progress_step_cb:
            progress_step_cb(pct)

    log("Starting…", "INFO", 0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using: {device}", "INFO")

    log("Reading frames…", "STEP", 10)
    frames = extract_frames(input_path)

    log("Computing CLIP features…", "STEP", 40)
    E = embed_frames(frames, device)

    log("Computing HSV features…", "STEP", 55)
    H = hsv_hist(frames)

    log("Building similarity…", "STEP", 65)
    S = combined_similarity(E, H)

    log("Flow refinement…", "STEP", 85)
    S = add_flow_to_similarity(frames, S)

    log("Spectral seed…", "STEP")
    seed = spectral_seriation(S)

    log("Beam search…", "STEP")
    order = beam_search(S, start_order=seed)

    log("2-opt…", "STEP")
    order = two_opt(order, S)

    log("Local repair…", "STEP")
    order = local_repair(order, S)

    if reverse_after:
        log("Reversing order…", "STEP")
        order = order[::-1]

    log("Saving output…", "STEP", 100)
    ordered_frames = [frames[i] for i in order]
    save_video(ordered_frames, output_path, fps=fps)

    log(f"Saved: {output_path}", "OK", 100)
    return output_path


# =========================================================
# GUI
# =========================================================
def run_gui():
    from PySide6.QtWidgets import (
        QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout,
        QFileDialog, QLabel, QCheckBox, QTextEdit, QMessageBox, QLineEdit, QProgressBar
    )
    from PySide6.QtGui import QTextCharFormat, QColor, QTextCursor
    from PySide6.QtCore import QThread, Signal
    import subprocess
    import platform
    import webbrowser

    def open_external_video(path):
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            try:
                subprocess.call(["xdg-open", path])
            except:
                webbrowser.open(path)

    class Worker(QThread):
        progress = Signal(str)
        progress_pct = Signal(int)
        done = Signal(str)
        failed = Signal(str)

        def __init__(self, in_path, out_path, rev):
            super().__init__()
            self.inp = in_path
            self.outp = out_path
            self.rev = rev

        def run(self):
            try:
                def cb(msg):
                    self.progress.emit(msg)

                def cb_pct(p):
                    self.progress_pct.emit(p)

                unjumble_then_optionally_reverse(
                    self.inp,
                    self.outp,
                    reverse_after=self.rev,
                    progress_cb=cb,
                    progress_step_cb=cb_pct
                )
                self.done.emit(self.outp)
            except Exception as e:
                self.failed.emit(str(e))

    class App(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Video Unjumbler")
            self.resize(950, 600)

            self.input_path = None

            layout = QVBoxLayout()
            ctrls = QHBoxLayout()

            self.btn_select = QPushButton("Select Input")
            self.btn_select.clicked.connect(self.pick_input)

            self.reverse_chk = QCheckBox("Reverse output")

            self.out_name = QLineEdit()
            self.out_name.setPlaceholderText("Output name (e.g. result.mp4)")

            self.btn_run = QPushButton("Run")
            self.btn_run.clicked.connect(self.run_it)

            play_in = QPushButton("Play Input")
            play_in.clicked.connect(self.play_input)

            play_out = QPushButton("Play Output")
            play_out.clicked.connect(self.play_output)

            ctrls.addWidget(self.btn_select)
            ctrls.addWidget(self.reverse_chk)
            ctrls.addWidget(QLabel("Output:"))
            ctrls.addWidget(self.out_name)
            ctrls.addWidget(self.btn_run)
            ctrls.addWidget(play_in)
            ctrls.addWidget(play_out)

            # === PROGRESS BAR ===
            self.bar = QProgressBar()
            self.bar.setRange(0, 100)
            self.bar.setValue(0)
            self.bar.hide()

            self.log = QTextEdit()
            self.log.setReadOnly(True)

            layout.addLayout(ctrls)
            layout.addWidget(self.bar)
            layout.addWidget(self.log)

            self.setLayout(layout)

            self.worker = None

        def color_text(self, txt):
            cursor = self.log.textCursor()
            fmt = QTextCharFormat()

            if "❌" in txt:
                fmt.setForeground(QColor("#ff3b30"))
            elif "✅" in txt:
                fmt.setForeground(QColor("#34c759"))
            elif "🔄" in txt:
                fmt.setForeground(QColor("#ffcc00"))
            else:
                fmt.setForeground(QColor("#ffffff"))

            cursor.insertText(txt + "\n", fmt)
            self.log.moveCursor(QTextCursor.End)

        def msg(self, txt):
            self.color_text(txt)

        def set_pct(self, p):
            self.bar.setValue(p)

        def pick_input(self):
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Pick Video",
                "",
                "Videos (*.mp4 *.avi)"
            )
            if path:
                self.input_path = path
                self.msg(pretty(f"Selected: {path}", "INFO"))

                if not self.out_name.text():
                    fn = os.path.splitext(os.path.basename(path))[0]
                    self.out_name.setText(fn + "_fixed.mp4")

        def play_input(self):
            if not self.input_path:
                self.msg(pretty("No input selected", "ERR"))
                return
            open_external_video(self.input_path)

        def play_output(self):
            name = self.out_of_name.text().strip()
            if not name:
                return
            path = os.path.abspath(os.path.join("output", name))
            if not os.path.exists(path):
                self.msg(pretty("Output not generated yet", "WARN"))
                return
            open_external_video(path)

        def run_it(self):
            if not self.input_path:
                QMessageBox.warning(self, "Missing input", "Select input first.")
                return

            name = self.out_name.text().strip()
            if not name:
                QMessageBox.warning(self, "Missing output", "Provide output filename.")
                return

            os.makedirs("output", exist_ok=True)
            out_path = os.path.abspath(os.path.join("output", name))
            rev = self.reverse_chk.isChecked()

            self.msg(pretty(f"Starting → {out_path}   Reverse={rev}", "INFO"))
            self.btn_run.setEnabled(False)
            self.bar.show()

            self.worker = Worker(self.input_path, out_path, rev)
            self.worker.progress.connect(self.msg)
            self.worker.progress_pct.connect(self.set_pct)
            self.worker.done.connect(self.done)
            self.worker.failed.connect(self.fail)
            self.worker.start()

        def done(self, out):
            self.msg(pretty("Done", "OK"))
            self.btn_run.setEnabled(True)
            self.bar.setValue(100)

        def fail(self, e):
            self.msg(pretty("ERROR: " + e, "ERR"))
            QMessageBox.critical(self, "Error", e)
            self.btn_run.setEnabled(True)
            self.bar.hide()

    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


# =========================================================
# CLI ENTRY
# =========================================================
def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="reconstructed.mp4")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    if args.gui:
        run_gui()
        return

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", args.output)

    unjumble_then_optionally_reverse(
        args.input,
        out_path,
        reverse_after=args.reverse
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_gui()
    else:
        main_cli()
