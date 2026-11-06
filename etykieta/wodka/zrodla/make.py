import numpy as np, cv2, math
from PIL import Image, ImageDraw, ImageFont
SRC = '/root/.claude/uploads/b3d44c30-0bd8-5a31-a38f-0863b18d98e5/f4cb082f-image.png'
base = cv2.imread(SRC); H, W = base.shape[:2]
COOK = 'fonts/UnifrakturCookwght700.ttf'; GRENZE = 'fonts/GrenzeGotisch900.ttf'; FELL = 'fonts/IMFellEnglishSC.ttf'
layers = {}
def layer(name):
    if name not in layers: layers[name] = Image.new('L', (W, H), 0)
    return layers[name], ImageDraw.Draw(layers[name])
def center(name, txt, size, cx, baseline, font, max_w=None):
    im, d = layer(name); f = ImageFont.truetype(font, size); w = d.textlength(txt, font=f)
    if max_w and w > max_w: return center(name, txt, int(size * max_w / w), cx, baseline, font)
    d.text((cx - w / 2, baseline), txt, font=f, fill=255, anchor='ls'); print(txt, size, int(w))
def left(name, txt, size, x, baseline, font, max_w=None):
    im, d = layer(name); f = ImageFont.truetype(font, size); w = d.textlength(txt, font=f)
    if max_w and w > max_w: return left(name, txt, int(size * max_w / w), x, baseline, font)
    d.text((x, baseline), txt, font=f, fill=255, anchor='ls'); print(txt, size, int(w))
def curve(name, txt, size, p0, p1, p2, font):
    im, d = layer(name); f = ImageFont.truetype(font, size)
    pt = lambda t: ((1-t)**2*p0[0]+2*(1-t)*t*p1[0]+t*t*p2[0], (1-t)**2*p0[1]+2*(1-t)*t*p1[1]+t*t*p2[1])
    pts = np.array([pt(t) for t in np.linspace(0, 1, 2001)]); seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    total = d.textlength(txt, font=f); s = (seg[-1] - total) / 2
    for ch in txt:
        adv = d.textlength(ch, font=f); i = min(max(np.searchsorted(seg, s + adv / 2), 1), len(seg) - 1)
        x, y = pts[i]; dx, dy = pts[i] - pts[i - 1]; ang = math.degrees(math.atan2(dy, dx))
        g = Image.new('L', (size * 3, size * 3), 0); ImageDraw.Draw(g).text((size * 1.5, size * 1.5), ch, font=f, fill=255, anchor='ms')
        g = g.rotate(-ang, resample=Image.BICUBIC, center=(size * 1.5, size * 1.5))
        nx, ny = -math.sin(math.radians(ang)), math.cos(math.radians(ang)); off = size * 0.33
        im.paste(255, (int(x + nx * off - size * 1.5), int(y + ny * off - size * 1.5)), g); s += adv

curve('ink', 'Ignis et Aqua', 56, (270, 638), (543, 732), (845, 645), COOK)
center('title', 'von Karma', 178, 543, 893, GRENZE, max_w=720)
center('red', 'Altdorfer Korn', 72, 543, 972, COOK, max_w=560)
left('ink', 'oppelt gebrannt zu Altdorf', 50, 408, 1031, COOK, max_w=486)
left('ink', 'am Hofe des Imperators', 50, 408, 1088, COOK, max_w=486)
center('ink', 'Anno Sigmari · MMDXXII', 36, 650, 1160, FELL)
im_, d_ = layer('gold'); fD = ImageFont.truetype(COOK, 178)
bb = d_.textbbox((0, 0), 'D', font=fD, anchor='ls')
d_.text((277 - (bb[0] + bb[2]) / 2, 1110 - (bb[1] + bb[3]) / 2), 'D', font=fD, fill=255, anchor='ls')

paper = base.astype(np.float32)
g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
hp = g - cv2.GaussianBlur(g, (0, 0), 3)
rng = np.random.default_rng(5)
n = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 6); n = (n - n.mean()) / n.std()
def A(name, blur=0.6, dil=True, strong=False):
    a = np.asarray(layers[name]).astype(np.float32) / 255
    if dil: a = cv2.dilate(a, np.ones((2, 2), np.uint8))
    a = cv2.GaussianBlur(a, (0, 0), blur)
    if strong: wear = np.clip(1 - np.maximum(hp - 14, 0) / 80, 0.85, 1)
    else: wear = np.clip(1 - np.maximum(hp - 8, 0) / 60, 0.7, 1) * np.clip(1 - np.maximum(n - 2.0, 0) * 0.6, 0.6, 1)
    return (a * wear)[..., None]
def ink_on(img, a, col, mix=0.15):
    col = np.array(col, np.float32)  # BGR
    inkpix = col * (1 - mix) + img * (col / 255) * mix
    return img * (1 - a) + inkpix * a

out = paper.copy()
# blue wash in the initial frame (hand colouring)
wash = np.zeros((H, W), np.float32); wash[1031:1189, 203:352] = 1
wn = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 10); wn = (wn - wn.min()) / (wn.max() - wn.min())
wash = cv2.GaussianBlur(wash, (0, 0), 1.5) * (0.75 + 0.25 * wn)
blue = np.array([120, 72, 44], np.float32)  # BGR, deep royal blue like the shield
out = out * (1 - wash[..., None] * 0.85) + (blue * 0.6 + out * blue / 255 * 0.4) * wash[..., None] * 0.85
# gold initial with dark outline
ag = np.asarray(layers['gold']).astype(np.float32) / 255
outline = cv2.GaussianBlur(cv2.dilate(ag, np.ones((5, 5), np.uint8)), (0, 0), 0.8)[..., None]
out = ink_on(out, outline, (22, 26, 34), 0.1)
grad = np.linspace(0, 1, H)[:, None, None]
goldc = (np.array([95, 190, 228], np.float32) * (1 - grad) + np.array([30, 110, 160], np.float32) * grad)
shine = (1 + np.clip(hp[..., None], -15, 15) / 90)
gold = np.clip(goldc * shine * (0.9 + 0.2 * wn[..., None]), 0, 255)
ga = cv2.GaussianBlur(ag, (0, 0), 0.7)[..., None]
out = out * (1 - ga) + gold * ga
# texts
out = ink_on(out, A('ink'), (24, 30, 38))
out = ink_on(out, A('title', strong=True), (20, 25, 33))
out = ink_on(out, A('red'), (38, 42, 140), 0.2)
out = np.clip(out, 0, 255).astype(np.uint8)
cv2.imwrite('wodka/final.png', out)
