"""Download pinned official contracts/workloads, never final submission code."""
import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

BASE = 'https://huggingface.co'
REPO = 'flashinfer-ai/mlsys26-contest'

def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read(), response.headers.get('Link', '')
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--decode-blobs', action='store_true')
    parser.add_argument('--prefill-blobs', action='store_true')
    parser.add_argument('--other-blobs', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root/'dataset-lock.json'
    if lock.exists():
        revision = json.loads(lock.read_text())['revision']
    else:
        revision = json.loads(fetch(f'{BASE}/api/datasets/{REPO}')[0])['sha']
        lock.write_text(json.dumps(dict(repo=REPO, revision=revision), indent=2))
    url = f'{BASE}/api/datasets/{REPO}/tree/{revision}?recursive=true&limit=1000'
    files = []
    while url:
        payload, link = fetch(url)
        files.extend(x for x in json.loads(payload) if x['type'] == 'file')
        url = next((part.split('>')[0].strip().lstrip('<') for part in link.split(',') if 'rel="next"' in part), None)
    selected = [x for x in files if x['path'].startswith(('definitions/', 'workloads/')) or (args.decode_blobs and x['path'].startswith('blob/workloads/gdn/gdn_decode_')) or (args.prefill_blobs and x['path'].startswith('blob/workloads/gdn/gdn_prefill_'))]
    if args.other_blobs:
        selected += [x for x in files if x['path'].startswith(('blob/workloads/dsa_paged/', 'blob/workloads/moe/'))]
    def download(item):
        path = root/item['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size != item['size']:
            content = fetch(f'{BASE}/datasets/{REPO}/resolve/{revision}/{item["path"]}')[0]
            if len(content) != item['size']:
                raise RuntimeError('size mismatch: '+item['path'])
            path.write_bytes(content)
        return dict(path=item['path'], size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(download, selected))
    (root/'download-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(revision=revision, downloaded=len(manifest), bytes=sum(x['size'] for x in manifest))))

if __name__ == '__main__':
    main()
