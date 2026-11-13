"""ABI-3 resident driver: one model upload and one DMA pair per image."""
import time
import numpy as np
try:
    from .resident import compile_model,MAGIC
except ImportError:
    from resident import compile_model,MAGIC

COUNTERS={0x0c:'core_cycles',0x10:'tiles',0x20:'dma_input_words',0x24:'dma_output_words',
          0x28:'macs',0x2c:'load_wait_cycles',0x30:'overlap_cycles',0x40:'network_cycles',
          0x44:'model_reads',0x48:'activation_reads',0x4c:'activation_write_bytes',0x50:'images'}
MONITOR={8:'hp_read_beats',12:'hp_write_beats',16:'hp_write_bytes',20:'hp_read_requests',24:'hp_write_requests'}

class ResidentAccelerator:
    def __init__(self,overlay,model,timeout=10):
        from pynq import allocate
        self.control=overlay.rs_accel_0.mmio
        if self.control.read(0)!=0x52530003: raise ValueError('resident inference requires ABI-3 bitstream')
        self.monitor=overlay.rs_ddr_monitor_0.mmio
        if self.monitor.read(0)!=0x5253444d: raise ValueError('DDR monitor ABI mismatch')
        self.dma=overlay.axi_dma_0;self.timeout=timeout
        self.model=compile_model(model)
        caps=self.control.read(0x18)
        self.p=caps&255;self.q=(caps>>8)&255;self.filters=(caps>>16)&255
        self.cols=(self.control.read(0x14)>>8)&255
        self.pes=self.filters*3*self.cols
        self.control.write(4,1);self.monitor.write(4,1)
        # Fixed buffers survive every inference. Upload uses the same TX buffer.
        self.tx=allocate(shape=(max(len(self.model.words)+16,784),),dtype=np.uint32)
        self.rx=allocate(shape=(4096,),dtype=np.uint32)
        self.failed=False
        before=self.counters()
        ack=self._transfer(self.model.upload_packet(),1)
        if ack[0]!=MAGIC or not self.control.read(0x54)&0x80000000: raise RuntimeError('model upload failed')
        self.upload_counters=self.delta(before,self.counters())

    def counters(self):
        values={name:int(self.control.read(addr)) for addr,name in COUNTERS.items()}
        values.update({name:int(self.monitor.read(addr)) for addr,name in MONITOR.items()})
        return values

    @staticmethod
    def delta(before,after):
        return {k:(int(after[k])-int(v))&0xffffffff for k,v in before.items()}

    def _transfer(self,packet,count):
        if self.failed: raise RuntimeError('previous transfer failed; reprogram overlay before reuse')
        n=len(packet)
        if n>len(self.tx) or count>len(self.rx): raise ValueError('resident DMA buffer capacity')
        self.tx[:n]=packet
        try:
            self.dma.recvchannel.transfer(self.rx,nbytes=count*4)
            self.dma.sendchannel.transfer(self.tx,nbytes=n*4)
            end=time.monotonic()+self.timeout
            while not(self.dma.sendchannel.idle and self.dma.recvchannel.idle):
                if self.control.read(8)&2: raise RuntimeError('resident hardware fault')
                if self.dma.sendchannel.error or self.dma.recvchannel.error: raise RuntimeError('DMA error')
                if time.monotonic()>end: raise TimeoutError('resident DMA timeout')
                time.sleep(0.0001)
            self.dma.sendchannel.wait();self.dma.recvchannel.wait()
            if self.dma.recvchannel.transferred!=count*4: raise RuntimeError('short resident output')
            self.rx.invalidate()
            return np.array(self.rx[:count],np.uint32,copy=True)
        except BaseException:
            self.failed=True
            raise

    def infer(self,x,layers=0):
        raw=self._transfer(self.model.input_packet(x,layers),self.model.output_count(layers))
        return self.model.decode_output(raw,layers)

    def close(self):
        if self.failed:
            # Keep allocated memory live until DMA is halted, even on an abort.
            self.dma.sendchannel.stop();self.dma.recvchannel.stop()
        self.tx.freebuffer();self.rx.freebuffer()


def main():
    import argparse,json
    try:
        from .protocol import load_vector
        from .pynq_board import Board
    except ImportError:
        from protocol import load_vector
        from pynq_board import Board
    ap=argparse.ArgumentParser(description='Run one embedded validation image with resident FPGA inference')
    ap.add_argument('bitstream');ap.add_argument('archive');ap.add_argument('--image',type=int,default=0)
    a=ap.parse_args()
    _,_,expected=load_vector(a.archive,a.image)
    board=Board(a.bitstream)
    accel=ResidentAccelerator(board.overlay,a.archive)
    try:
        before=accel.counters();start=time.monotonic()
        actual=accel.infer(expected['input'])
        elapsed=time.monotonic()-start
        np.testing.assert_array_equal(actual,expected['logits'])
        print('PASS exact resident logits',json.dumps(dict(image=a.image,prediction=int(actual.argmax()),logits=actual.tolist(),host_seconds=elapsed,counters=accel.delta(before,accel.counters()))))
    finally:accel.close()

if __name__=='__main__':main()
