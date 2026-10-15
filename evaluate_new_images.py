"""Uji model pada gambar baru di luar dataset.

Susun gambar sebagai test_images/<nama_kelas>/<file>.jpg (nama folder = label asli).
Folder bernama 'unknown' boleh dipakai untuk gambar di luar 10 kelas.

Jalankan:  python evaluate_new_images.py
Hasil:     results.csv + ringkasan akurasi di terminal.
"""
import csv
from pathlib import Path

from PIL import Image

from inference import ROOT, load_config, load_model, predict

EXTS = {".jpg", ".jpeg", ".png"}


def main():
    config = load_config()
    model = load_model(config)
    rows = []
    for path in sorted((ROOT / "test_images").rglob("*")):
        if path.suffix.lower() not in EXTS:
            continue
        true_label = path.parent.name
        top3 = predict(model, Image.open(path), config, top_k=3)
        pred, conf = top3[0]
        rows.append(
            {
                "file": path.relative_to(ROOT).as_posix(),
                "label_asli": true_label,
                "prediksi": pred,
                "confidence": f"{conf:.4f}",
                "benar": pred == true_label,
                "top3": "; ".join(f"{c} {p:.2%}" for c, p in top3),
            }
        )

    if not rows:
        print("Tidak ada gambar di test_images/<kelas>/.")
        return

    out = ROOT / "results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        mark = "OK " if r["benar"] else "X  "
        print(f"{mark}{r['file']:45s} -> {r['prediksi']:10s} ({float(r['confidence']):.1%})")

    labeled = [r for r in rows if r["label_asli"] in config["class_names"]]
    if labeled:
        correct = sum(r["benar"] for r in labeled)
        print(f"\nAkurasi gambar baru: {correct}/{len(labeled)} = {correct / len(labeled):.1%}")
    print(f"Hasil ditulis ke {out}")


if __name__ == "__main__":
    main()
