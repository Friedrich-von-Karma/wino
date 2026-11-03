"""Składa szafa-cyberpunk.html z src.html i zdjęć z img/.

Zdjęcia: img/<id>.jpg (wersja premium) i img/<id>-alt.jpg (tańszy zamiennik).
Obsługiwane: .jpg .jpeg .png .webp. Są wklejane do pliku jako data URI,
więc gotowa strona działa bez internetu.
Użycie: python3 build.py [ścieżka_do_fragmentu_artefaktu]
"""
import base64, json, pathlib, sys

here = pathlib.Path(__file__).parent
src = (here / "src.html").read_text(encoding="utf-8")
mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
imgs = {}
for f in sorted((here / "img").glob("*")):
    if f.suffix.lower() in mime:
        imgs[f.stem] = f"data:{mime[f.suffix.lower()]};base64," + base64.b64encode(f.read_bytes()).decode()
frag = src.replace("/*IMG_MAP*/{}", json.dumps(imgs))

head, body = frag.split("</style>\n", 1)
full = ('<!DOCTYPE html>\n<html lang="pl">\n<head>\n<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + head + "</style>\n</head>\n<body>\n" + body + "</body>\n</html>\n")
(here / "szafa-cyberpunk.html").write_text(full, encoding="utf-8")
if len(sys.argv) > 1:
    pathlib.Path(sys.argv[1]).write_text(frag, encoding="utf-8")
print(f"zdjęć: {len(imgs)}")
