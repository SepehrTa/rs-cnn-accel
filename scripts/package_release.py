"""Package source, matching deployment files, trained weights and evidence."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--board',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--float',dest='float_model',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('build/rs-cnn-accel-pynq-z2-validated.zip'))
    args = ap.parse_args()
    out = (ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT/'build'):
        ap.error('--out must remain inside repository build/')
    files = {}
    # Package versioned inputs, excluding GUI session state, caches and local
    # incremental checkpoints even if Vivado created them inside .srcs/.
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in tracked:
        path=ROOT/name
        if name and path.is_file() and not (name.startswith('deployment/') and path.suffix=='.zip'):files[name]=path
    files['rs_accel.bit'] = ROOT/args.board/'rs_accel.bit'
    files['rs_accel.hwh'] = ROOT/args.board/'rs_accel.hwh'
    files['resnet8_int8.npz'] = ROOT/args.model
    files['resnet8_float.npz'] = ROOT/args.float_model
    # Read everything before opening the output; missing artifacts fail clearly.
    data = {name:path.read_bytes() for name,path in sorted(files.items())}
    manifest = {name:dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest())
                for name,value in data.items()}
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,value in data.items():
            archive.writestr(name,value)
        archive.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    # Verify stored bytes against the same manifest after closing the archive.
    with zipfile.ZipFile(out) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('release archive CRC failure')
        for name,entry in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != entry['sha256']:
                raise RuntimeError(f'release archive hash mismatch: {name}')
    print(out)
    print('PASS release',len(data),'files,',out.stat().st_size,'bytes; SHA256',hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
