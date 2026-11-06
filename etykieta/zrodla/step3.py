import numpy as np, cv2, math
from PIL import Image, ImageDraw, ImageFont
base = cv2.imread('clean2.png')
H, W = base.shape[:2]
COOK = 'fonts/UnifrakturCookwght700.ttf'
alpha = Image.new('L', (W, H), 0)
dr = ImageDraw.Draw(alpha)

def text_center(txt, size, cx, baseline, font=COOK, max_w=None):
    f = ImageFont.truetype(font, size)
    w = dr.textlength(txt, font=f)
    if max_w and w > max_w:
        return text_center(txt, int(size * max_w / w), cx, baseline, font)
    dr.text((cx - w / 2, baseline), txt, font=f, fill=255, anchor='ls')
    return f, w

def text_left(txt, size, x, baseline, font=COOK):
    f = ImageFont.truetype(font, size)
    dr.text((x, baseline), txt, font=f, fill=255, anchor='ls')
    return dr.textlength(txt, font=f)

def text_on_curve(txt, size, p0, p1, p2, font=COOK):
    f = ImageFont.truetype(font, size)
    def pt(t):
        return ((1-t)**2*p0[0]+2*(1-t)*t*p1[0]+t*t*p2[0], (1-t)**2*p0[1]+2*(1-t)*t*p1[1]+t*t*p2[1])
    ts = np.linspace(0, 1, 2001); pts = np.array([pt(t) for t in ts])
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    total = dr.textlength(txt, font=f)
    s = (seg[-1] - total) / 2
    for ch in txt:
        adv = dr.textlength(ch, font=f)
        mid = s + adv / 2
        i = np.searchsorted(seg, mid); i = min(max(i, 1), len(seg) - 1)
        x, y = pts[i]; dx, dy = pts[i] - pts[i - 1]
        ang = math.degrees(math.atan2(dy, dx))
        g = Image.new('L', (size * 3, size * 3), 0)
        ImageDraw.Draw(g).text((size * 1.5, size * 1.5), ch, font=f, fill=255, anchor='ms')
        g = g.rotate(-ang, resample=Image.BICUBIC, center=(size * 1.5, size * 1.5))
        # baseline sits slightly below the curve so letters are centred on the band
        nx, ny = -math.sin(math.radians(ang)), math.cos(math.radians(ang))
        off = size * 0.33
        alpha.paste(255, (int(x + nx * off - size * 1.5), int(y + ny * off - size * 1.5)), g)
        s += adv

# title (keeps the original "Wein" underneath)
alpha_body = alpha
alpha = Image.new('L', (W, H), 0); dr = ImageDraw.Draw(alpha)
text_center('von Karma', 185, 543, 903, font='fonts/GrenzeGotisch900.ttf', max_w=800)
alpha_title = alpha
alpha = alpha_body; dr = ImageDraw.Draw(alpha)
# text beside the illuminated R: reads "Rotwein aus Talabheim / im Auge des Waldes"
text_left('otwein aus Talabheim', 56, 362, 1160)
text_left('im Auge des Waldes', 56, 362, 1222)
# date
text_center('Anno Sigmari · MMDXXII', 38, 543, 1293, font='fonts/IMFellEnglishSC.ttf')
# ribbon motto
text_on_curve('In Silva Veritas', 50, (240, 640), (543, 728), (846, 636))
alpha.save('alpha.png')

a = np.asarray(alpha).astype(np.float32) / 255
a = cv2.dilate(a, np.ones((2, 2), np.uint8))
at = cv2.GaussianBlur(np.asarray(alpha_title).astype(np.float32) / 255, (0, 0), 0.6)
a = cv2.GaussianBlur(a, (0, 0), 0.6)
# wear: paper grain lifts ink, plus a few worn patches
g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
hp = g - cv2.GaussianBlur(g, (0, 0), 3)
rng = np.random.default_rng(7)
n = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 6); n = (n - n.mean()) / n.std()
fine = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 1.2); fine = (fine - fine.mean()) / fine.std()
wear = np.clip(1 - np.maximum(hp - 8, 0) / 60, 0.7, 1) * np.clip(1 - np.maximum(n - 2.0, 0) * 0.6, 0.6, 1) * np.clip(1 - np.maximum(fine - 2.4, 0)*0.7, 0.6, 1)
a = a * wear
wear_t = np.clip(1 - np.maximum(hp - 14, 0) / 80, 0.85, 1)
a = np.maximum(a, at * wear_t)
# ink colour taken from the original "Wein"
reg = base[930:1045, 400:710].reshape(-1, 3)
ink = np.median(reg[reg.sum(1) < 150], axis=0).astype(np.float32)
paper = base.astype(np.float32)
inkpix = ink * 0.8 + paper * (ink / 255) * 0.15
out = paper * (1 - a[..., None]) + inkpix * a[..., None]
cv2.imwrite('final.png', np.clip(out, 0, 255).astype(np.uint8))
print('ink', ink)
