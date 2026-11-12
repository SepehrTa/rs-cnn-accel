"""Fetch a pinned official PYNQ PS7 preset into this repository's build folder.

No installation/global board repository changes. Only the generated property
dictionary is extracted; the remote Tcl program is never executed.
"""
import hashlib
from pathlib import Path
import re
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
URL='https://raw.githubusercontent.com/Xilinx/PYNQ/v3.1/boards/Pynq-Z2/petalinux_bsp/hardware_project/pynqz2.tcl'
SHA256='ebcf4641c5eecba0c0017fd7616215fa7a050de53e12b3bb59bec0115145ebaa'


def main():
    target=ROOT/'build'/'vendor'
    target.mkdir(parents=True,exist_ok=True)
    raw=urllib.request.urlopen(URL,timeout=30).read()
    if hashlib.sha256(raw).hexdigest()!=SHA256:
        raise RuntimeError('upstream preset changed; inspect before accepting a new hash')
    text=raw.decode('utf-8')
    match=re.search(r'set_property -dict \[list (.*?)\] \$ps7',text,re.S)
    if not match or 'MT41J256M16 RE-125' not in match.group(1):
        raise RuntimeError('unexpected official preset format')
    # All entries must be literal CONFIG.<property> {value} lines.
    entries=re.findall(r'CONFIG\.[A-Za-z0-9_]+\s+\{[^{}\n]*\}',match.group(1))
    if len(entries)<100:
        raise RuntimeError('incomplete PS7 preset')
    header='# Copyright (C) 2022 Xilinx, Inc\n# SPDX-License-Identifier: BSD-3-Clause\n'
    header+=f'# Extracted from {URL}\n# Source SHA256: {hashlib.sha256(raw).hexdigest()}\n'
    (target/'pynq_z2_ps.tcl').write_text(header+'set_property -dict [list \\\n'+' \\\n'.join(entries)+' \\\n] [get_bd_cells ps7]\n')
    (target/'pynqz2.tcl').write_bytes(raw)
    (target/'SOURCE.txt').write_text(URL+'\nSHA256 '+hashlib.sha256(raw).hexdigest()+'\n')
    print(f'Fetched {len(entries)} PS7 properties into {target}')


if __name__=='__main__': main()
