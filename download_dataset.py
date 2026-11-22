"""Download and verify the pinned public AMdataset original photographs."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from http.client import IncompleteRead
from pathlib import Path
from time import sleep
from urllib.parse import quote
from urllib.request import Request, urlopen

REVISION = "52fe75dec70c1e8046493b4e3256f8f752a4d39d"
REPOSITORY = "mahsan2/AMdataset"


def fetch(url):
    for delay in (0, 1, 3):
        sleep(delay)
        try:
            with urlopen(Request(url, headers={"User-Agent": "IDAT7215-data-download"}), timeout=45) as response:
                return response.read()
        except (OSError, IncompleteRead) as error:
            last_error = error
    raise last_error


def blob_hash(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def download(blob):
    path = Path("data") / blob["path"]
    if path.is_file() and blob_hash(path.read_bytes()) == blob["sha"]:
        return
    content = fetch(f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{quote(blob['path'])}")
    if blob_hash(content) != blob["sha"]:
        raise ValueError(f"Git blob checksum mismatch for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def main():
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=true"))
    blobs = [blob for blob in tree["tree"] if blob["type"] == "blob" and blob["path"].startswith("dataset/")]
    if tree["truncated"] or len(blobs) != 271:
        raise ValueError("Unexpected file listing for the pinned original dataset.")
    with ThreadPoolExecutor(max_workers=14) as executor:
        for count, _ in enumerate(executor.map(download, blobs), 1):
            if count % 25 == 0 or count == len(blobs):
                print(f"Verified {count}/{len(blobs)} original images", flush=True)


if __name__ == "__main__":
    main()
