# Paralaks galeri

Tek bir fotoğraf ya da resimden, bakış açısına göre derinlik kazanan "pencere" görüntüleri.
Her eser fareyle, telefonu eğerek ya da kamerayla kafa takibiyle izlenebilir.

## Yayına alma (GitHub Pages)

1. GitHub'da yeni bir depo aç (örn. `paralaks-galeri`), bu klasörün içeriğini yükle.
   Web arayüzünden "Add file → Upload files" ile sürükle-bırak yeterli.
2. Depoda **Settings → Pages** bölümüne gir. "Source" olarak **Deploy from a branch**,
   dal olarak `main`, klasör olarak `/ (root)` seç, kaydet.
3. Bir iki dakika sonra galeri `https://KULLANICIADI.github.io/paralaks-galeri/` adresinde açılır.

Kamera ve telefon eğim sensörü yalnızca HTTPS'te çalışır; GitHub Pages bunu kendiliğinden sağlar.

## Yeni eser ekleme

```bash
pip install onnxruntime opencv-python pillow numpy
python araclar/katmanla.py foto.jpg --galeri . --baslik "Başlık" --aciklama "Kişi / yer, yıl"
```

Bu komut `eserler/` altına görüntüleyiciyi, `kucuk/` altına küçük resmi ekler,
`galeri.json`'u günceller ve `index.html`'i yeniden üretir. Sonra değişiklikleri depoya yükle.

- İlk çalıştırmada derinlik modeli (~390 MB) `araclar/` içine iner; `.gitignore` sayesinde depoya girmez.
- Galeri başlığını ve açıklamasını `galeri.json` içinden değiştirip
  `python araclar/katmanla.py --galeri . --sadece-index` ile sayfayı yenile.
- Eser silmek için ilgili dosyaları sil, `galeri.json`'dan kaydını çıkar, index'i yeniden üret.
- Ayarlar: `--katman` (varsayılan 5), `--genislik` (varsayılan 1600 px).

## Gizlilik

GitHub Pages sitesi, depo özel olsa bile linki bilen herkese açıktır (Enterprise hesaplar hariç).
Paylaşılmasını istemediğin aile fotoğraflarını galeriye koyma; onları tek HTML dosyası olarak doğrudan gönder.

## Bilinen sınırlar

- Arkada kalan bölgeler OpenCV inpainting ile dolduruluyor; geniş açılarda kenarlarda bulanıklık görünür.
  `araclar/katmanla.py` içindeki `inpaint_color` fonksiyonu LaMa veya SD/FLUX inpainting ile değiştirilerek iyileştirilebilir.
- Kamera modu tek kişiyi izler; etki monoküler hareket paralaksıdır.
