"""Download the real ShanghaiT2DM dataset (CC BY 4.0) from its official Figshare record.

    python data/download_real_data.py

Zhao Q. et al. "Chinese diabetes datasets for data-driven machine learning",
Scientific Data 10, 35 (2023). https://doi.org/10.6084/m9.figshare.c.6310860
"""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = "https://ndownloader.figshare.com/files/42966622"
MD5 = "4bfb61cfa506b48155fd9e841ee48e21"
OUT = Path(__file__).parent / "real_shanghai_raw"


def main():
    OUT.mkdir(exist_ok=True)
    zpath = OUT / "diabetes_datasets.zip"
    if not zpath.exists():
        print("Downloading ShanghaiT2DM (~3.7 MB) ...")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        zpath.write_bytes(urllib.request.urlopen(req).read())
    if hashlib.md5(zpath.read_bytes()).hexdigest() != MD5:
        raise SystemExit("Checksum mismatch - delete the zip and try again.")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(OUT, [n for n in z.namelist()
                           if not n.startswith("__MACOSX") and n.endswith((".xls", ".xlsx", ".txt"))])
    print("Ready:", OUT)


if __name__ == "__main__":
    main()
