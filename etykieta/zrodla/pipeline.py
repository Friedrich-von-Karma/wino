import numpy as np, cv2
SRC = '/root/.claude/uploads/b3d44c30-0bd8-5a31-a38f-0863b18d98e5/6454d6fb-image.png'
src = cv2.imread(SRC); H, W = src.shape[:2]
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV); gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
yy, xx = np.mgrid[0:H, 0:W]
K = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))

# ---------- text to remove ----------
dark = (gray < 105).astype(np.uint8)
tm = np.zeros((H, W), np.uint8)
for x0, y0, x1, y1 in [(150, 770, 960, 912), (150, 905, 385, 945), (725, 905, 960, 960), (385, 1110, 720, 1232), (350, 1262, 740, 1310)]:
    tm[y0:y1, x0:x1] |= dark[y0:y1, x0:x1]
tm = cv2.dilate(cv2.morphologyEx(tm, cv2.MORPH_CLOSE, K(25)), K(17))
tm[905:1060, 385:725] = 0

# ---------- seal ----------
bx0, by0, bx1, by1 = 770, 1085, 1080, 1400
sub = hsv[by0:by1, bx0:bx1]
redp = (((sub[..., 0] < 14) | (sub[..., 0] > 160)) & (sub[..., 1] > 90) & (sub[..., 2] > 35)).astype(np.uint8)
redp = cv2.morphologyEx(redp, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
cnts, _ = cv2.findContours(redp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
seal = np.zeros((H, W), np.uint8)
cv2.fillPoly(seal, [cv2.convexHull(max(cnts, key=cv2.contourArea)) + np.array([bx0, by0])], 1)
seal = cv2.dilate(seal, K(31))
orn = np.zeros((H, W), np.uint8)
d110 = (gray < 110).astype(np.uint8)
orn[1000:1262, 860:976] = d110[1000:1262, 860:976]
orn = cv2.dilate(cv2.morphologyEx(orn, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)), np.ones((11, 11), np.uint8))
region = (seal == 1) | (orn == 1)
mir_zone = region & ((yy >= 1262) | (xx >= 976))
seal_fill = region & ~mir_zone

img = src.copy()
sx = np.clip(np.round(1081.5 - xx).astype(int), 0, W - 1)
m = cv2.GaussianBlur(mir_zone.astype(np.float32), (0, 0), 3)[..., None]
img = (src[yy, sx] * m + img * (1 - m)).astype(np.uint8)

zone = ((tm == 1) | seal_fill).astype(np.uint8)
zone = cv2.dilate(zone, K(7))

# ---------- clean parchment samples from the original ----------
lab = cv2.connectedComponentsWithStats((gray < 80).astype(np.uint8), 8)
n, labels, stats, _ = lab
bigc = np.isin(labels, np.nonzero(stats[:, cv2.CC_STAT_AREA] > 400)[0][1:]).astype(np.uint8)
inkstruct = cv2.dilate(bigc, K(7))
redorn = ((((hsv[..., 0] < 7) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60))).astype(np.uint8)
redorn = cv2.dilate(redorn, K(9))
banned = inkstruct | redorn | cv2.dilate(zone, K(21)) | cv2.dilate(region.astype(np.uint8), K(21))
interior = np.zeros((H, W), np.uint8); interior[110:1340, 110:975] = 1
ok = (banned == 0) & (interior == 1)
T = 28
print("ok frac", ok.mean(), "ink", inkstruct.mean(), "red", redorn.mean(), "zone", zone.mean())
ii = cv2.integral((~ok).astype(np.uint8))
cands = []
for y in range(110, 1340 - T, 4):
    for x in range(110, 975 - T, 4):
        if ii[y + T, x + T] - ii[y, x + T] - ii[y + T, x] + ii[y, x] == 0:
            cands.append((y, x))
print('samples', len(cands))
rng = np.random.default_rng(3)

# ---------- quilt tiles over the zone ----------
ys, xs = np.nonzero(zone)
y0, y1, x0, x1 = ys.min() - T, ys.max() + T, xs.min() - T, xs.max() + T
acc = np.zeros((H, W, 3), np.float32); wacc = np.zeros((H, W), np.float32)
win = np.outer(np.hanning(T + 2)[1:-1], np.hanning(T + 2)[1:-1]).astype(np.float32) + 1e-3
step = T // 2
for ty in range(max(y0, 0), min(y1, H - T), step):
    for tx in range(max(x0, 0), min(x1, W - T), step):
        if zone[ty:ty + T, tx:tx + T].max() == 0:
            continue
        cy, cx = cands[rng.integers(len(cands))]
        tile = src[cy:cy + T, cx:cx + T].astype(np.float32)
        if rng.random() < .5: tile = tile[:, ::-1]
        acc[ty:ty + T, tx:tx + T] += tile * win[..., None]
        wacc[ty:ty + T, tx:tx + T] += win
comp = acc / np.maximum(wacc, 1e-6)[..., None]
# match low frequencies to the surroundings
w = ((zone == 0) & (inkstruct == 0) & (redorn == 0)).astype(np.float32)
num = cv2.GaussianBlur(img.astype(np.float32) * w[..., None], (0, 0), 30); den = cv2.GaussianBlur(w, (0, 0), 30)[..., None]
cm = np.median(src[ok].reshape(-1, 3), axis=0).astype(np.float32)
low = (num + cm * 0.05) / (den + 0.05)
cw = (wacc > 0).astype(np.float32)
cnum = cv2.GaussianBlur(comp * cw[..., None], (0, 0), 30); cden = cv2.GaussianBlur(cw, (0, 0), 30)[..., None]
clow = cnum / np.maximum(cden, 1e-3)
fillv = comp - clow + low
soft = cv2.GaussianBlur(zone.astype(np.float32), (0, 0), 3)[..., None]
out = np.clip(fillv * soft + img * (1 - soft), 0, 255).astype(np.uint8)
cv2.imwrite('clean2.png', out)
