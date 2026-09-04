"""Download one representative image per ImageNet class for each confusable
class pair used in the SHAP case study.

Source: EliSchwartz/imagenet-sample-images (one public-domain-licensed sample
image per ImageNet-1k class). Images are cached in data/images/ and skipped
if already present, so re-running is cheap.
"""

import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "data" / "images"
BASE_URL = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master"

# (output_name, source_filename_in_repo)
# Grouped by confusable pair/trio; group name is used in results reporting.
CLASS_PAIRS = {
    "dogs_husky_malamute": [
        ("husky", "n02110185_Siberian_husky.JPEG"),
        ("malamute", "n02110063_malamute.JPEG"),
    ],
    "cats_tabby_egyptian": [
        ("tabby_cat", "n02123045_tabby.JPEG"),
        ("egyptian_cat", "n02124075_Egyptian_cat.JPEG"),
        ("tiger_cat", "n02123159_tiger_cat.JPEG"),
    ],
    "birds_jay_magpie": [
        ("jay", "n01580077_jay.JPEG"),
        ("magpie", "n01582220_magpie.JPEG"),
    ],
    "sharks": [
        ("great_white_shark", "n01484850_great_white_shark.JPEG"),
        ("tiger_shark", "n01491361_tiger_shark.JPEG"),
    ],
    "big_cats": [
        ("leopard", "n02128385_leopard.JPEG"),
        ("jaguar", "n02128925_jaguar.JPEG"),
    ],
    "foxes": [
        ("red_fox", "n02119022_red_fox.JPEG"),
        ("kit_fox", "n02119789_kit_fox.JPEG"),
        ("grey_fox", "n02120505_grey_fox.JPEG"),
    ],
    "elephants": [
        ("indian_elephant", "n02504013_Indian_elephant.JPEG"),
        ("african_elephant", "n02504458_African_elephant.JPEG"),
    ],
    "bears": [
        ("american_black_bear", "n02133161_American_black_bear.JPEG"),
        ("brown_bear", "n02132136_brown_bear.JPEG"),
    ],
    "terriers": [
        ("yorkshire_terrier", "n02094433_Yorkshire_terrier.JPEG"),
        ("silky_terrier", "n02097658_silky_terrier.JPEG"),
    ],
    "retrievers": [
        ("golden_retriever", "n02099601_golden_retriever.JPEG"),
        ("labrador_retriever", "n02099712_Labrador_retriever.JPEG"),
    ],
}


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    for group, items in CLASS_PAIRS.items():
        for name, filename in items:
            dest = IMAGES_DIR / f"{name}.jpg"
            if dest.exists():
                skipped += 1
                continue
            url = f"{BASE_URL}/{filename}"
            urllib.request.urlretrieve(url, dest)
            downloaded += 1
            print(f"[{group}] downloaded {name}.jpg")
    print(f"Done. Downloaded {downloaded}, skipped {skipped} (already present).")


if __name__ == "__main__":
    main()
