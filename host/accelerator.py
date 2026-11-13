"""PYNQ DMA driver. Importable on a development PC; PYNQ is imported on use."""
import argparse
import time

import numpy as np

try:
    from .protocol import forward, load_model, load_vector
    from .packed import pack_packet, output_words as packed_output_words, unpack_output
except ImportError:
    from protocol import forward, load_model, load_vector
    from packed import pack_packet, output_words as packed_output_words, unpack_output


class Accelerator:
    ABI = 0x52530002

    def __init__(self, overlay, dma_name='axi_dma_0', control_name='rs_accel_0', timeout=10.0):
        self.dma = getattr(overlay, dma_name)
        self.control = getattr(overlay, control_name).mmio
        self.timeout = timeout
        self.input_words = self.output_words = self.commands = 0
        abi=self.control.read(0)
        if abi not in (0x52530001,0x52530002,0x52530003):
            raise ValueError('accelerator ABI mismatch; bitstream and driver must match')
        self.packed=abi>=0x52530002
        caps=self.control.read(0x14)
        self.cols=(caps>>8)&255
        self.filters=(caps>>24)&255
        if caps & 0x00ff00ff != 0x00030020 or not 1 <= self.cols <= 32 or not 1 <= self.filters <=16:
            raise ValueError('unsupported accelerator dimensions')
        self.control.write(4, 1)

    def __call__(self, packet, output_words):
        from pynq import allocate
        packet = np.asarray(packet, dtype=np.uint32)
        semantic_packet=packet
        physical_outputs=packed_output_words(packet,output_words) if self.packed else output_words
        if self.packed: packet=pack_packet(packet)
        if not 0 < packet.nbytes < 2**23 or not 0 < physical_outputs*4 < 2**23:
            raise ValueError('packet exceeds configured DMA length register')
        tx = allocate(shape=(len(packet),), dtype=np.uint32)
        rx = allocate(shape=(physical_outputs,), dtype=np.int32)
        failed = True
        try:
            tx[:] = packet
            # Arm receive first: some commands emit while MM2S still sends data.
            self.dma.recvchannel.transfer(rx)
            self.dma.sendchannel.transfer(tx)
            end = time.monotonic()+self.timeout
            while not (self.dma.sendchannel.idle and self.dma.recvchannel.idle):
                if self.control.read(8) & 2:
                    raise RuntimeError('accelerator rejected packet header')
                if self.dma.sendchannel.error or self.dma.recvchannel.error:
                    raise RuntimeError('AXI DMA error')
                if time.monotonic() > end:
                    raise TimeoutError('DMA timeout; reset DMA and accelerator before reuse')
                time.sleep(0.0001)
            # Now nonblocking: let PYNQ retire active buffers and record actual
            # received byte counts using its standard DMA completion path.
            self.dma.sendchannel.wait()
            self.dma.recvchannel.wait()
            if self.dma.recvchannel.transferred != physical_outputs*4:
                raise RuntimeError('DMA received a short accelerator packet')
            rx.invalidate()
            result = np.array(rx, dtype=np.int32, copy=True)
            if self.packed: result=unpack_output(semantic_packet,result,output_words)
            self.input_words += len(packet)
            self.output_words += physical_outputs
            self.commands += 1
            failed = False
            return result
        finally:
            # Never free a live DMA buffer on an exceptional path. Stop channels
            # first so an in-flight transfer cannot write freed physical memory.
            if failed:
                self.dma.sendchannel.stop()
                self.dma.recvchannel.stop()
            tx.freebuffer()
            rx.freebuffer()

    def infer(self, model, x):
        q, graph = load_model(model)
        return forward(self, q, graph, x, cols=self.cols,filters=self.filters)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('bitstream')
    ap.add_argument('model')
    ap.add_argument('--image',type=int,default=0)
    args=ap.parse_args()
    # Check selection and archive integrity before Board loads/programs the PL.
    q,graph,expected_tensors=load_vector(args.model,args.image)
    from pynq_board import Board
    board=Board(args.bitstream)
    accel=Accelerator(board.overlay)
    start=time.monotonic()
    t=forward(accel,q,graph,expected_tensors['input'],cols=accel.cols,filters=accel.filters)
    elapsed=time.monotonic()-start
    for name,value in t.items():
        if not np.array_equal(value,expected_tensors[name]):
            where=np.argwhere(value!=expected_tensors[name])[0]
            raise AssertionError(f'{name}: first mismatch {tuple(where)}')
    print('PASS every tensor; image',args.image,'class',int(t['logits'].argmax()),'host seconds',elapsed)
    print('busy cycles',accel.control.read(12),'commands',accel.commands,
          'stream bytes',4*(accel.input_words+accel.output_words))


if __name__=='__main__': main()
