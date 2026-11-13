"""Generate labelled analytical/simulation/board comparison without conflating units."""
import argparse,dataclasses,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'mapper'),str(ROOT/'model')]
from rs_model import HwConfig,best_mapping,layers_from_graph,Mapping,count
from graph import resnet8_graph

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--board',type=Path)
    ap.add_argument('--simulation',type=Path)
    ap.add_argument('--out',type=Path,default=ROOT/'build'/'resident-comparison.json')
    a=ap.parse_args();hw=HwConfig();layers=[]
    for l in layers_from_graph(resnet8_graph()):
        m,c=best_mapping(l,hw,'cycles')
        layers.append(dict(layer=l.name,mapping=dataclasses.asdict(m),**c))
    fixed=[]
    for layer in layers_from_graph(resnet8_graph()):
        mapping=Mapping(e=min(14,layer.E),p=2,q=2,r=1,t=4)
        fixed.append(dict(layer=layer.name,mapping=dataclasses.asdict(mapping),**count(layer,mapping,hw)))
    fixed_totals={k:sum(v[k] for v in fixed) for k in ('mac','cycles','dram','glb')}
    fixed_totals['utilization']=fixed_totals['mac']/(hw.pes*fixed_totals['cycles'])
    total={k:sum(l[k] for l in layers) for k in ('mac','cycles','dram','glb','dram_weight','dram_ifmap','dram_ofmap')}
    total['utilization']=total['mac']/(hw.pes*total['cycles'])
    total['dram_scalar_accesses_per_mac']=total['dram']/total['mac']
    report=dict(analytical=dict(scope='convolutions only, per-layer DDR residency, unconstrained mapper schedule; scalar access counts are not AXI beat counts',hardware=dataclasses.asdict(hw),layers=layers,totals=total),
                resident_transport_prediction=dict(input_bytes=3136,output_bytes=40,model_upload_bytes=82492,model_upload_output_bytes=4,hp_read_beats_per_image=392,hp_write_beats_per_image=5,scope='64-bit HP0 DMA interface, no DMA prefetch beyond command; excludes CPU/cache and DRAM row commands'))
    report['fixed_mapping_compute']=dict(scope='convolution-only analytical count at e<=14,p=2,q=2,r=1,t=4; excludes actual finite-port/cache stalls, scalar output reduction/requantization and nonconvolution work',layers=fixed,totals=fixed_totals)
    if a.simulation:
        s=a.simulation.read_text()
        line=next(l for l in s.splitlines() if 'PASS RESIDENT' in l)
        values={k:int(v) for k,v in re.findall(r'(\w+)=\s*(\d+)',line)}
        images=values['layers']//14
        report['simulation']=dict(scope='complete resident network RTL, stalled output, selected trained image repeated without model reload',images=images,**values,network_cycles_per_image=values['cycles']/images,utilization=values['macs']/(hw.pes*values['cycles']))
    if a.board:
        b=json.loads(a.board.read_text())
        if b['status']!='PASS':raise ValueError('board run did not pass')
        n=b['completed'];t=b['totals']
        if len(b['runs'])!=n or t['macs']!=12500992*n:raise ValueError('incomplete board evidence')
        report['board']=dict(scope='whole network on actual PYNQ-Z2; exact test logits checked per image',images=n,accuracy=b['accuracy'],correct=b['correct'],mean_host_seconds=b['mean_host_seconds'],network_cycles_per_image=b['mean_network_cycles'],utilization=b['utilization_macs_per_pe_cycle'],hp_beats_per_mac=b['hp_beats_per_mac'],hp_bytes_per_mac=b['hp_bytes_per_mac'],axis_bytes_per_image=b['axis_bytes_per_image'],upload_counters=b['upload_counters'],totals=t,bitstream_sha256=b['bitstream_sha256'])
        old=json.loads((ROOT/'docs/reports/board-2025.2/trained-inference.json').read_text())
        report['v1_comparison']=dict(scope='same trained model; v1 timed four embedded images repeated3, v2 timed labelled test set; different sample populations',v1_mean_host_seconds=old['mean_host_seconds'],host_speedup=old['mean_host_seconds']/b['mean_host_seconds'],axis_byte_reduction=1-b['axis_bytes_per_image']/8437392)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='analytical'},indent=2))
    print('mapper totals',total,flush=True)
if __name__=='__main__':main()
