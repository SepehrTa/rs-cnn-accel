"""Verify resident intermediates, then execute CIFAR-10 test images on the FPGA.

Reports accepted PS HP0 DMA beats/bytes separately from AXIS payload bytes.
Model upload is measured separately and reused throughout the labelled run.
"""
import argparse,hashlib,json,pickle,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'host'),str(ROOT/'model')]
from protocol import load_vector
from resident_accelerator import ResidentAccelerator

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bitstream',type=Path);ap.add_argument('archive',type=Path)
    ap.add_argument('test_batch',type=Path)
    ap.add_argument('--count',type=int,default=10000)
    ap.add_argument('--report',type=Path,default=Path('resident-board.json'))
    ap.add_argument('--skip-intermediates',action='store_true')
    a=ap.parse_args()
    with a.test_batch.open('rb') as f: batch=pickle.load(f,encoding='bytes')
    images=np.asarray(batch[b'data'],np.uint8).reshape(-1,3,32,32)
    labels=np.asarray(batch[b'labels'],np.int64)
    with np.load(a.archive,allow_pickle=False) as z:
        golden=z['test_logits'].astype(np.int32)
        embedded=len(z['vec.input'])
    if not 1<=a.count<=len(images) or golden.shape!=(10000,10): raise ValueError('test set/golden shape mismatch')
    from pynq_board import Board
    import pynq
    board=Board(str(a.bitstream));accel=ResidentAccelerator(board.overlay,a.archive)
    report=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),pynq=pynq.__version__,numpy=np.__version__,
                bitstream_sha256=sha(a.bitstream),hwh_sha256=sha(a.bitstream.with_suffix('.hwh')),
                archive_sha256=sha(a.archive),test_batch_sha256=sha(a.test_batch),
                physical_pes=accel.pes,cols=accel.cols,filter_groups=accel.filters,p=accel.p,q=accel.q,
                clock_hz=pynq.Clocks.fclk0_mhz*1000000,model_words=len(accel.model.words),upload_counters=accel.upload_counters,
                debug_runs=[],runs=[])
    def save():
        temporary=a.report.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(a.report)
    save()
    try:
        if not a.skip_intermediates:
            for image in range(embedded):
                _,graph,t=load_vector(a.archive,image)
                exact=0
                for limit,l in enumerate(graph,1):
                    y=accel.infer(t['input'],limit)
                    np.testing.assert_array_equal(y,t[l['dst']],err_msg=f'embedded image {image} {l["dst"]}')
                    exact+=y.size
                report['debug_runs'].append(dict(image=image,exact_values=exact,layers=len(graph)))
                print('PASS INTERMEDIATES',image,exact,flush=True);save()
        report['dataset_started_utc']=datetime.now(timezone.utc).isoformat()
        start_dataset=time.monotonic();totals={};correct=0
        for i in range(a.count):
            x=(images[i].astype(np.int16)-128).astype(np.int8)
            before=accel.counters();start=time.monotonic()
            y=accel.infer(x).reshape(10)
            elapsed=time.monotonic()-start
            delta=accel.delta(before,accel.counters())
            np.testing.assert_array_equal(y,golden[i],err_msg=f'test image {i} logits')
            if delta['macs']!=accel.model.macs or delta['images']!=1 or delta['tiles']!=71:
                raise AssertionError(f'image {i}: operation counters {delta}')
            if delta['dma_input_words']!=784 or delta['dma_output_words']!=10:
                raise AssertionError('inference touched unexpected DMA data')
            predicted=int(y.argmax());correct+=predicted==int(labels[i])
            for k,v in delta.items(): totals[k]=totals.get(k,0)+v
            report['runs'].append(dict(image=i,label=int(labels[i]),prediction=predicted,logits=y.tolist(),host_seconds=elapsed,**delta))
            if (i+1)%100==0 or i+1==a.count:
                report.update(completed=i+1,correct=correct,accuracy=correct/(i+1),totals=totals)
                save();print('PASS DATASET',i+1,'correct',correct,'seconds',time.monotonic()-start_dataset,flush=True)
        report['dataset_seconds']=time.monotonic()-start_dataset
        report['mean_host_seconds']=float(np.mean([r['host_seconds'] for r in report['runs']]))
        report['mean_network_cycles']=totals['network_cycles']/a.count
        report['utilization_macs_per_pe_cycle']=totals['macs']/(accel.pes*totals['network_cycles'])
        report['hp_beats_per_mac']=(totals['hp_read_beats']+totals['hp_write_beats'])/totals['macs']
        report['hp_bytes_per_mac']=(8*totals['hp_read_beats']+totals['hp_write_bytes'])/totals['macs']
        report['axis_bytes_per_image']=4*(totals['dma_input_words']+totals['dma_output_words'])/a.count
        report['status']='PASS'
    except BaseException as e:
        report['status']='FAIL';report['error']=f'{type(e).__name__}: {e}';raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        save();accel.close()
    print('PASS RESIDENT BOARD',a.report,flush=True)
if __name__=='__main__': main()
