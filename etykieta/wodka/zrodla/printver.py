import sys, numpy as np, cv2
from PIL import Image
src, out_png, out_pdf = sys.argv[1:4]
def medf(v, k):
    p = np.pad(v, k // 2, mode='edge'); return np.median(np.lib.stride_tricks.sliding_window_view(p, k), axis=1)
im = cv2.imread(src); H, W = im.shape[:2]
g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (0, 0), 2); pp = g >= 100
def first_run(arr, run=8):
    c = np.convolve(arr.astype(int), np.ones(run, int), 'valid'); i = np.argmax(c >= run)
    return i if c[i] >= run else None
L = np.zeros(H); R = np.zeros(H); T = np.zeros(W); B = np.zeros(W)
for y in range(H):
    a = first_run(pp[y]); b = first_run(pp[y][::-1]); L[y] = a if a is not None else W; R[y] = W-1-b if b is not None else -1
for x in range(W):
    a = first_run(pp[:, x]); b = first_run(pp[:, x][::-1]); T[x] = a if a is not None else H; B[x] = H-1-b if b is not None else -1
fix = lambda v, inw: np.minimum(v, medf(v, 41) + 5) if inw else np.maximum(v, medf(v, 41) - 5)
L, T, R, B = fix(L, 1), fix(T, 1), fix(R, 0), fix(B, 0)
yy, xx = np.mgrid[0:H, 0:W]
m = ((xx >= L[:, None]) & (xx <= R[:, None]) & (yy >= T[None, :]) & (yy <= B[None, :])).astype(np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)[..., None]
pr = (im * a + 255 * (1 - a)).astype(np.uint8)
p = Image.fromarray(cv2.cvtColor(pr, cv2.COLOR_BGR2RGB))
p.save(out_png, dpi=(306, 306)); p.save(out_pdf, resolution=W / (90 / 25.4))
p.resize((W // 2, H // 2)).save('wodka/pr_small.png')
