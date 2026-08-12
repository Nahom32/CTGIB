import argparse
import hashlib
import json
import os
import sys
import urllib.request
import zipfile

ZENODO_RECORD = "7213796"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

DATASETS = [
    "Contacts",
    "reddit",
    "TG_network_datasets",
    "lastfm",
    "USLegis",
    "wikipedia",
    "UNtrade",
    "mooc",
    "SocialEvo",
    "enron",
    "Flights",
    "uci",
    "UNvote",
    "CanParl",
]


def fetch_file_list():
    with urllib.request.urlopen(ZENODO_API) as resp:
        record = json.load(resp)
    return {f["key"]: f for f in record["files"]}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    if os.path.exists(dest):
        print(f"{dest} already exists, skipping download")
        return True
    print(f"Downloading {url} -> {dest}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{100 * done // total}%", end="", flush=True)
    print()
    os.replace(tmp, dest)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download datasets for TGIB experiments")
    parser.add_argument("name", nargs="?", help="dataset name, e.g. wikipedia, reddit, USLegis")
    parser.add_argument("--list", action="store_true", help="list available datasets")
    parser.add_argument("--out", default="./processed", help="output directory")
    args = parser.parse_args()

    if args.list:
        for d in DATASETS:
            print(d)
        return

    if not args.name:
        parser.print_help()
        sys.exit(1)

    if args.name not in DATASETS:
        print(f"Unknown dataset '{args.name}'. Available datasets:")
        for d in DATASETS:
            print(f"  {d}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    files = fetch_file_list()
    key = f"{args.name}.zip"
    info = files.get(key)
    if not info:
        print(f"No file named {key} found in the record")
        sys.exit(1)

    zip_path = os.path.join(args.out, key)
    download(info["links"]["self"], zip_path)

    if md5(zip_path) != info["checksum"].split(":", 1)[1]:
        print("Checksum verification failed")
        sys.exit(1)
    print("Checksum OK")

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.out)
    print(f"Extracted into {args.out}")


if __name__ == "__main__":
    main()
