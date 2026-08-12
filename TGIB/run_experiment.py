import argparse
import os
import subprocess
import sys

NON_BIPARTITE = {"USLegis", "CanParl"}


def ensure_dirs():
    for d in ["log", "saved_models", "saved_checkpoints"]:
        os.makedirs(d, exist_ok=True)


def download(data):
    print(f"[pipeline] Downloading {data}")
    subprocess.run([sys.executable, "download_data.py", data], check=True)


def preprocess(data):
    print(f"[pipeline] Preprocessing {data}")
    import process

    bipartite = data not in NON_BIPARTITE
    process.run(data, bipartite=bipartite)


def train(data, extra_args):
    print(f"[pipeline] Training on {data}")
    cmd = [sys.executable, "-u", "learn_edge.py", "-d", data] + extra_args
    print("[pipeline] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="End-to-end TGIB experiment runner")
    parser.add_argument("-d", "--data", type=str, default="wikipedia",
                        help="dataset name, e.g. wikipedia, reddit, USLegis")
    parser.add_argument("--skip-download", action="store_true",
                        help="skip downloading if raw csv exists")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="skip preprocessing (use existing ml_* files)")
    parser.add_argument("--skip-train", action="store_true",
                        help="only download/preprocess, do not train")
    parser.add_argument("--trainer-args", nargs=argparse.REMAINDER,
                        help="extra args passed to learn_edge.py, e.g. -- --bs 200 --uniform")
    args, unknown = parser.parse_known_args()

    raw_csv = f"./processed/{args.data}/{args.data}.csv"
    ml_csv = f"./processed/ml_{args.data}.csv"

    ensure_dirs()

    if not args.skip_download:
        download(args.data)

    if not os.path.exists(raw_csv):
        print(f"[pipeline] {raw_csv} not found after download, extract it under ./processed/")
        sys.exit(1)

    if not args.skip_preprocess:
        preprocess(args.data)

    if not os.path.exists(ml_csv):
        print(f"[pipeline] {ml_csv} not found after preprocessing")
        sys.exit(1)

    if not args.skip_train:
        extra = list(args.trainer_args or []) + unknown
        if extra and extra[0] == "--":
            extra = extra[1:]
        train(args.data, extra)

    print("[pipeline] Done")


if __name__ == "__main__":
    main()
