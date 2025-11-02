import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import clip


# ============================
# BASIC I/O
# ============================
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
        raise ValueError("No frames read. Check --input path/codec.")
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
        raise RuntimeError("VideoWriter failed to open. Try different extension/codecs.")

    for f in frames:
        writer.write(f)
    writer.release()


# ============================
# FEATURES (CLIP + HSV)
# ============================
def downscale_frame(f, max_side=640):
    h, w = f.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return f
    nh, nw = int(h*s), int(w*s)
    return cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)


def embed_frames(frames, device, max_side=640, model_name="ViT-B/32"):
    model, preprocess = clip.load(model_name, device=device)
    embs = []
    for f in tqdm(frames, desc="Embedding (CLIP)"):
        f2 = downscale_frame(f, max_side=max_side)
        img = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
        tens = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            e = model.encode_image(tens).float()
            e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
            e = e.cpu().numpy()[0]
        embs.append(e)
    return np.vstack(embs)


def hsv_hist(frames, bins=(32,32,8), max_side=480):
    hists = []
    for f in tqdm(frames, desc="Computing HSV hist"):
        f2 = downscale_frame(f, max_side=max_side)
        hsv = cv2.cvtColor(f2, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256]).astype(np.float32)
        hist = hist.flatten()
        hist /= (np.linalg.norm(hist)+1e-8)
        hists.append(hist)
    return np.vstack(hists)


# ============================
# SIMILARITY
# ============================
def cosine_sim(A):
    A = A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-8)
    return A @ A.T


def mutual_knn_bonus(S, k=10, bonus=0.15):
    N = S.shape[0]
    idx = np.argsort(-S,axis=1)[:,:k]
    mask = np.zeros_like(S,dtype=np.float32)
    for i in range(N):
        nbrs = idx[i]
        for j in nbrs:
            if i in idx[j]:
                mask[i,j] = bonus
    np.fill_diagonal(mask,0.)
    return mask


def combined_similarity(clip_emb, hsv_emb, w_clip=0.70, w_hsv=0.30,
                        mknn_k=10, mknn_bonus=0.15):
    S_clip = cosine_sim(clip_emb)
    S_hsv  = cosine_sim(hsv_emb)
    S = w_clip * S_clip + w_hsv * S_hsv
    S += mutual_knn_bonus(S, k=mknn_k, bonus=mknn_bonus)
    np.fill_diagonal(S, -1.0)
    return S


# ============================
# OPTICAL FLOW
# ============================
def warp_with_flow(img, flow):
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    mx = (gx + flow[...,0]).astype(np.float32)
    my = (gy + flow[...,1]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def flow_residual(a,b):
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    try:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(a_gray,b_gray,None)
    except:
        flow = cv2.calcOpticalFlowFarneback(a_gray,b_gray,None,
                                            0.5,3,25,3,5,1.2,0)
    a_warp = warp_with_flow(a_gray, flow)
    diff = (a_warp.astype(np.float32)-b_gray.astype(np.float32))
    return float(np.mean(np.abs(diff))/255.)


def add_flow_to_similarity(frames, S, topk=40, side=384, w_flow=0.35):
    N = len(frames)
    sm = [downscale_frame(f, max_side=side) for f in frames]
    S2 = S.copy()
    idx = np.argsort(-S,axis=1)[:,:topk]
    for i in tqdm(range(N),desc="Flow refine"):
        for j in idx[i]:
            if i==j: continue
            a,b = sm[i],sm[j]
            res = flow_residual(a,b)
            s2 = np.exp(-5.0*res)
            S2[i,j] = (1-w_flow)*S2[i,j] + w_flow*s2
    return S2


# ============================
# ORDER SEARCH
# ============================
def spectral_seriation(S):
    D = 1. - np.maximum(S,-1.)
    D -= D.min()
    D = (D + D.T)*0.5
    W = np.exp(-D/(np.mean(D)+1e-8))
    np.fill_diagonal(W,0.)
    d = np.sum(W,axis=1)
    L = np.diag(d) - W
    vals,vecs = np.linalg.eigh(L)
    if len(vals)<2:
        return list(range(len(S)))
    fiedler = vecs[:,1]
    return np.argsort(fiedler).tolist()


def path_score(order,S):
    return float(sum(S[order[i],order[i+1]] for i in range(len(order)-1)))


def beam_search(S, start_order=None, beam_width=14, look_k=32):
    N = S.shape[0]
    if start_order is None:
        start = int(np.argmax(S.sum(axis=1)))
        beams = [([start], {start})]
    else:
        beams = [(start_order[:1],{start_order[0]})]

    for _ in range(1,N):
        new = []
        for path,used in beams:
            last = path[-1]
            cand = np.argsort(-S[last])[:look_k]
            cnt=0
            for c in cand:
                if c in used: continue
                newp = path+[c]
                newu = set(used)
                newu.add(c)
                sc = path_score(newp,S)
                new.append((newp,newu,sc))
                cnt+=1
                if cnt >= beam_width*2:
                    break
        if not new:
            break

        new.sort(key=lambda x:x[2],reverse=True)
        beams = [(p,u) for (p,u,_) in new[:beam_width]]

    best = max(beams,key=lambda bu:(len(bu[0]),path_score(bu[0],S)))[0]

    remain = list(set(range(N))-set(best))
    cur = best[-1]
    while remain:
        i = int(np.argmax(S[cur,remain]))
        nxt = remain[i]
        best.append(nxt)
        remain.remove(nxt)
        cur = nxt
    return best


def two_opt(order,S,iters=300):
    n=len(order)
    best=order[:]
    for _ in range(iters):
        improved=False
        for i in range(1,n-2):
            a,b = best[i-1],best[i]
            for j in range(i+1,n-1):
                c,d = best[j],best[j+1]
                before=S[a,b]+S[c,d]
                after =S[a,c]+S[b,d]
                if after>before:
                    best[i:j+1]=reversed(best[i:j+1])
                    improved=True
        if not improved:
            break
    return best


def local_repair(order,S,window=6,rounds=2):
    n=len(order)
    arr=order[:]
    for _ in range(rounds):
        changed=False
        for start in range(0,n-window+1):
            seg = arr[start:start+window]
            base = path_score(seg,S)
            best_seg=seg[:]
            best_sc=base
            for i in range(len(seg)):
                for j in range(i+1,len(seg)):
                    c2=seg[:]
                    c2[i],c2[j]=c2[j],c2[i]
                    sc=path_score(c2,S)
                    if sc>best_sc:
                        best_sc=sc
                        best_seg=c2[:]
            if best_seg!=seg:
                arr[start:start+window]=best_seg
                changed=True
        if not changed:
            break
    return arr


# ============================
# MAIN
# ============================
def unjumble(input_path, output_path, reverse_after=False, fps=30):
    print(">> Extract frames")
    frames = extract_frames(input_path)

    print(">> CLIP features")
    device="cuda" if torch.cuda.is_available() else "cpu"
    E = embed_frames(frames,device)

    print(">> HSV features")
    H = hsv_hist(frames)

    print(">> Similarity")
    S = combined_similarity(E,H)

    print(">> Flow refine")
    S = add_flow_to_similarity(frames,S)

    print(">> Spectral seriation")
    seed = spectral_seriation(S)

    print(">> Beam search")
    order = beam_search(S,start_order=seed)

    print(">> 2-opt")
    order = two_opt(order,S)

    print(">> Local repair")
    order = local_repair(order,S)

    if reverse_after:
        print(">> Reversing")
        order = order[::-1]

    out_frames = [frames[i] for i in order]

    print(">> Saving video:", output_path)
    save_video(out_frames, output_path, fps=fps)
    print("✅ Done.")


# ============================
# CLI ENTRY
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reconstructed.mp4")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    out_full = os.path.abspath(os.path.join("output", args.output))

    unjumble(args.input, out_full, reverse_after=args.reverse, fps=args.fps)


if __name__ == "__main__":
    main()
