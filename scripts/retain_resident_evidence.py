"""Independently validate and retain the full 10,000-image hardware evidence."""
import argparse,gzip,hashlib,json,pickle,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--test-batch',type=Path,required=True)
    ap.add_argument('--build-dir',type=Path,required=True)
    ap.add_argument('--simulation',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=ROOT/'docs/reports/resident-v2')
    a=ap.parse_args();raw=a.report.read_bytes();r=json.loads(raw)
    if r['status']!='PASS' or r['completed']!=10000 or len(r['runs'])!=10000:
        raise ValueError('a completed PASS on all 10,000 board images is required')
    if [v['image'] for v in r['runs']]!=list(range(10000)):raise ValueError('missing/reordered images')
    with np.load(a.archive,allow_pickle=False) as z:gold=z['test_logits'].astype(np.int32)
    actual=np.asarray([v['logits'] for v in r['runs']],np.int32)
    np.testing.assert_array_equal(actual,gold)
    with a.test_batch.open('rb') as f:labels=np.asarray(pickle.load(f,encoding='bytes')[b'labels'])
    np.testing.assert_array_equal([v['label'] for v in r['runs']],labels)
    np.testing.assert_array_equal([v['prediction'] for v in r['runs']],actual.argmax(1))
    for v in r['runs']:
        if (v['macs'],v['images'],v['dma_input_words'],v['dma_output_words'],v['tiles'])!=(12500992,1,784,10,71):
            raise ValueError(f'image {v["image"]} counter inconsistency')
    if len(r['debug_runs'])!=4 or any((v['layers'],v['exact_values'])!=(14,114762) for v in r['debug_runs']):
        raise ValueError('four complete board intermediate comparisons required')
    if r['bitstream_sha256']!=sha((a.build_dir/'rs_accel.bit').read_bytes()):raise ValueError('bitstream differs')
    if r['hwh_sha256']!=sha((a.build_dir/'rs_accel.hwh').read_bytes()):raise ValueError('HWH differs')
    if r['archive_sha256']!=sha(a.archive.read_bytes()) or r['test_batch_sha256']!=sha(a.test_batch.read_bytes()):raise ValueError('dataset/model differs')
    for k,v in r['totals'].items():
        if v!=sum(row[k] for row in r['runs']):raise ValueError(f'total counter {k} differs')
    correct=int((actual.argmax(1)==labels).sum())
    if r['correct']!=correct or r['accuracy']!=correct/10000:raise ValueError('reported accuracy differs')
    summary={k:v for k,v in r.items() if k!='runs'}
    summary['exact_test_logits']=int(actual.size)
    summary['independent_correct']=int((actual.argmax(1)==labels).sum())
    summary['raw_report_sha256']=sha(raw)
    summary['counter_ranges']={k:dict(min=min(v[k] for v in r['runs']),max=max(v[k] for v in r['runs'])) for k in r['totals']}
    a.out.mkdir(parents=True,exist_ok=True)
    packed=gzip.compress(raw,mtime=0)
    (a.out/'full-board-report.json.gz').write_bytes(packed)
    summary['compressed_report_sha256']=sha(packed)
    (a.out/'board-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    for name in ['timing.rpt','utilization.rpt','drc.rpt']:
        shutil.copyfile(a.build_dir/name,a.out/name)
    shutil.copyfile(a.simulation,a.out/'resident-xsim.txt')
    sources={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for folder in ['hw','host','sim','model','scripts'] for p in (ROOT/folder).glob('*') if p.is_file()}
    for name,relative in [('board-design','rs_board.srcs/sources_1/bd/system/system.bd'),('board-wrapper','rs_board.gen/sources_1/bd/system/hdl/system_wrapper.v')]:
        sources[name]=sha((a.build_dir/relative).read_bytes())
    (a.out/'source-sha256.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8')
    print('PASS independent evidence validation: 10,000 ordered images, 100,000 exact logits,',summary['independent_correct'],'correct;',len(packed),'compressed bytes')
if __name__=='__main__':main()
