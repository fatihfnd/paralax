#!/usr/bin/env python3
"""
katmanla.py — tek bir fotoğraftan/resimden kafa takipli paralaks görüntüleyici üretir.

Kullanım:
    pip install onnxruntime opencv-python pillow numpy
    python katmanla.py resim.jpg                                  # tek dosya: resim_3b.html
    python katmanla.py resim.jpg --galeri .. --baslik "Başlık" --aciklama "Sanatçı, yıl"
                                                                   # galeriye ekler, index.html'i yeniler

Adımlar:
  1) Depth Anything V2 (ONNX) ile derinlik haritası (yatay çevirme ile ortalama)
  2) Derinliğe göre katmanlara bölme
  3) Her katmanın arkasında kalan bölgeyi inpainting ile doldurma
     (burada OpenCV Telea; daha iyi sonuç için bu adım LaMa / SD inpainting ile değiştirilebilir)
  4) Her katman için kendi derinlik yüzeyi (mesh deplasmanı için)
  5) Hepsini tek bir HTML dosyasına gömme
"""
import argparse, base64, html, io, json, os, re, sys, urllib.request
import numpy as np, cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
MODEL_URL = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vitb.onnx"
HERE = os.path.dirname(os.path.abspath(__file__))


def load_image(path, max_w):
    im = Image.open(path)
    if im.width > max_w * 2:
        im.draft("RGB", (im.width // 2, im.height // 2))
    im = im.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    return np.asarray(im)[:, :, ::-1].copy()  # BGR


def estimate_depth(bgr, model_path):
    import onnxruntime as ort
    if not os.path.exists(model_path):
        print("Derinlik modeli indiriliyor (~390 MB)…")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    def run(im):
        x = cv2.resize(im, (518, 518), interpolation=cv2.INTER_CUBIC)
        x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        return sess.run(None, {name: x})[0][0]

    d = (run(rgb) + run(rgb[:, ::-1].copy())[:, ::-1]) / 2
    d = cv2.resize(d, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    return (d - d.min()) / (d.max() - d.min())  # 1 = yakın, 0 = uzak


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def inpaint_color(im, mask):
    s = 0.5
    sm = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mm = cv2.resize(mask, (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST)
    f = cv2.inpaint(sm, mm, 9, cv2.INPAINT_TELEA)
    f = cv2.GaussianBlur(f, (0, 0), 2)
    f = cv2.resize(f, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
    out = im.copy()
    out[mask > 0] = f[mask > 0]
    return out


def fill(val, keep):
    """Sınırlı (taşmayan) ekstrapolasyon: çok ölçekli normalize konvolüsyon."""
    m = keep.astype(np.float32)
    out = np.where(keep, val, 0).astype(np.float32)
    have = keep.copy()
    for sg in [1.5, 3, 6, 12, 24, 48, 96]:
        num = cv2.GaussianBlur(val * m, (0, 0), sg)
        den = cv2.GaussianBlur(m, (0, 0), sg)
        ok = (den > 1e-3) & ~have
        out[ok] = num[ok] / den[ok]
        have |= ok
    out[~have] = val[keep].mean() if keep.any() else 0.5
    return out


def pad_color(a, py, px):
    p = cv2.copyMakeBorder(a, py, py, px, px, cv2.BORDER_REFLECT_101)
    blur = cv2.GaussianBlur(p, (0, 0), 18)
    mask = np.ones(p.shape[:2], np.float32)
    mask[py:py + a.shape[0], px:px + a.shape[1]] = 0
    mask = cv2.GaussianBlur(mask, (0, 0), 6)[..., None]
    return (p * (1 - mask) + blur * mask * 0.7).astype(np.uint8)


def to_datauri(arr, fmt, **kw):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt, **kw)
    mime = {"WEBP": "image/webp", "PNG": "image/png"}[fmt]
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def build(path, n_layers, width, model_path, title):
    img = load_image(path, width)
    H, W = img.shape[:2]
    print(f"Görüntü {W}x{H}, derinlik hesaplanıyor…")
    v = estimate_depth(img, model_path).astype(np.float32)

    # katman eşikleri: yakın tarafta daha sık (paralaks orada daha belirgin)
    qs = np.linspace(0, 1, n_layers + 1)[1:-1]
    th = np.quantile(v, qs ** 0.8)
    band = np.digitize(v, th)              # 0 = en uzak … n-1 = en yakın
    t = 1 - v                              # 0 = yakın, 1 = uzak

    PAD = 0.14
    DW = 320
    DH = round(H * DW / W)
    py, px = round(H * PAD), round(W * PAD)
    dpy, dpx = round(DH * PAD), round(DW * PAD)
    ts = cv2.resize(t, (DW, DH), interpolation=cv2.INTER_AREA)

    layers = []
    for i in range(n_layers):
        own = (band == i).astype(np.uint8)
        nearer = (band > i).astype(np.uint8)
        col = inpaint_color(img, cv2.dilate(nearer, disk(14)) * 255) if nearer.any() else img.copy()
        if i == 0:
            a = np.ones((H, W), np.float32)
        else:
            a = np.maximum(own, cv2.dilate(own, disk(5)) * nearer).astype(np.float32)
            a = cv2.GaussianBlur(a, (0, 0), 1.6)
        keep = cv2.resize(own, (DW, DH), interpolation=cv2.INTER_NEAREST) > 0
        keep = cv2.erode(keep.astype(np.uint8), disk(1)) > 0
        if not keep.any():
            keep = cv2.resize(own, (DW, DH), interpolation=cv2.INTER_NEAREST) > 0
        dl = cv2.GaussianBlur(fill(ts, keep), (0, 0), 1.0)

        colp = pad_color(col, py, px)
        ap = cv2.copyMakeBorder(a, py, py, px, px, cv2.BORDER_REFLECT_101)
        rgba = np.dstack([colp[:, :, ::-1], (np.clip(ap, 0, 1) * 255).astype(np.uint8)])
        dp = cv2.copyMakeBorder(dl, dpy, dpy, dpx, dpx, cv2.BORDER_REPLICATE)
        dp8 = (np.clip(dp, 0, 1) * 65535).astype(np.uint16)
        layers.append(dict(
            img=to_datauri(rgba, "WEBP", quality=86, method=6),
            # 16 bit derinlik 2 kanala bölünür (R = yüksek bayt, G = düşük bayt)
            depth=to_datauri(np.dstack([(dp8 >> 8).astype(np.uint8), (dp8 & 255).astype(np.uint8),
                                        np.zeros_like(dp8, np.uint8)]), "PNG"),
        ))
        print(f"  katman {i}: alan %{own.mean() * 100:.1f}")

    data = dict(W=W, H=H, pad=PAD, title=title, layers=layers)
    return data, img


def render_viewer(data, title, sub, back_href=None):
    tpl = open(os.path.join(HERE, "viewer_template.html"), encoding="utf-8").read()
    back = f'<a id="back" href="{html.escape(back_href)}">Galeriye dön</a>' if back_href else ""
    return (tpl.replace("/*__DATA__*/null", json.dumps(data))
               .replace("__BACK__", back)
               .replace("__TITLE__", html.escape(title))
               .replace("__SUB__", html.escape(sub or "")))


TR = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def slugify(t):
    t = t.translate(TR).lower()
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-") or "eser"


def write_gallery(root):
    meta_p = os.path.join(root, "galeri.json")
    meta = json.load(open(meta_p, encoding="utf-8"))
    cards = []
    for e in meta["eserler"]:
        slug, title = e["slug"], html.escape(e["baslik"])
        sub = "<em>%s</em>" % html.escape(e["aciklama"]) if e.get("aciklama") else ""
        cards.append(
            '    <a class="work" href="eserler/%s.html">\n'
            '      <span class="frame"><img src="kucuk/%s.webp" alt="%s" width="%d" height="%d" loading="lazy"></span>\n'
            '      <span class="label"><strong>%s</strong>%s</span>\n'
            '    </a>' % (slug, slug, title, e["w"], e["h"], title, sub))
    tpl = open(os.path.join(HERE, "galeri_template.html"), encoding="utf-8").read()
    out = (tpl.replace("__GALERI_BASLIK__", html.escape(meta.get("baslik", "Paralaks galeri")))
              .replace("<p>__GALERI_ACIKLAMA__</p>",
                       "<p>%s</p>" % html.escape(meta["aciklama"]) if meta.get("aciklama") else "")
              .replace("<!--__ESERLER__-->", "\n".join(cards)))
    open(os.path.join(root, "index.html"), "w", encoding="utf-8").write(out)


def add_to_gallery(root, slug, title, sub, data, img):
    os.makedirs(os.path.join(root, "eserler"), exist_ok=True)
    os.makedirs(os.path.join(root, "kucuk"), exist_ok=True)
    open(os.path.join(root, "eserler", slug + ".html"), "w", encoding="utf-8").write(
        render_viewer(data, title, sub, back_href="../index.html"))
    th = Image.fromarray(img[:, :, ::-1].copy())
    th.thumbnail((900, 900), Image.LANCZOS)
    th.save(os.path.join(root, "kucuk", slug + ".webp"), "WEBP", quality=82, method=6)
    meta_p = os.path.join(root, "galeri.json")
    meta = json.load(open(meta_p, encoding="utf-8")) if os.path.exists(meta_p) else \
        dict(baslik="Paralaks galeri", aciklama="", eserler=[])
    entry = dict(slug=slug, baslik=title, aciklama=sub or "", w=th.width, h=th.height)
    meta["eserler"] = [e for e in meta["eserler"] if e["slug"] != slug] + [entry]
    json.dump(meta, open(meta_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    write_gallery(root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tek görüntüden kafa takipli paralaks görüntüleyici üretir.")
    ap.add_argument("resim", nargs="?", help="girdi görüntüsü")
    ap.add_argument("-o", "--cikti", help="tek dosya modu: çıktı HTML yolu")
    ap.add_argument("--galeri", help="galeri klasörü (örn. ..): eser eklenir, index.html yeniden üretilir")
    ap.add_argument("--baslik", help="eser başlığı")
    ap.add_argument("--aciklama", default="", help="başlığın altındaki satır (sanatçı, tarih vb.)")
    ap.add_argument("--slug", help="dosya adı (varsayılan: başlıktan)")
    ap.add_argument("--katman", type=int, default=5)
    ap.add_argument("--genislik", type=int, default=1600)
    ap.add_argument("--model", default=os.path.join(HERE, "depth_anything_v2_vitb.onnx"))
    ap.add_argument("--sadece-index", action="store_true", help="galeri.json'dan index.html'i yeniden üret")
    a = ap.parse_args()
    if a.sadece_index:
        write_gallery(a.galeri or "."); sys.exit()
    if not a.resim:
        ap.error("girdi görüntüsü gerekli")
    base = os.path.splitext(os.path.basename(a.resim))[0]
    title = a.baslik or base
    data, img = build(a.resim, a.katman, a.genislik, a.model, title)
    if a.galeri:
        slug = a.slug or slugify(title)
        add_to_gallery(a.galeri, slug, title, a.aciklama, data, img)
        print(f"Galeriye eklendi: eserler/{slug}.html")
    else:
        out = a.cikti or f"{base}_3b.html"
        open(out, "w", encoding="utf-8").write(render_viewer(data, title, a.aciklama))
        print(f"Yazıldı: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
