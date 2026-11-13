"""ABI-2 four-byte INT8 AXIS packing. Header/INT32 fields stay whole words.

Each independent byte segment is padded to a whole little-endian stream word.
Header word 15 bit 0 enables packing. Legacy semantic packets remain unchanged.
"""
import numpy as np


def segments(header):
    h=[int(v) for v in header]
    op=h[0]
    if op==0:
        k,u,pw,rows,nf,nc=(h[i] for i in (1,2,3,5,6,7))
        if nf>2: yield False,3*(nf-2)
        yield True,nc*(((rows-1)*u+k)*pw+nf*k*k)
    elif op==1: yield True,2*h[1]
    elif op==2: yield True,h[1]*h[2]
    elif op==3:
        yield False,h[1]
        for _ in range(h[2]):
            yield False,1
            yield True,h[1]
    else:
        raise ValueError('unsupported opcode for packed tile')


def pack_packet(packet):
    packet=np.asarray(packet,dtype=np.uint32)
    if packet.ndim!=1 or len(packet)<16 or packet[15]&1:
        raise ValueError('expected an unpacked semantic packet')
    header=packet[:16].copy(); header[15]|=1
    out=[header]; pos=16
    for is_byte,n in segments(header):
        values=packet[pos:pos+n]
        if len(values)!=n or n<=0: raise ValueError('invalid payload length')
        pos+=n
        if is_byte:
            signed=values.view(np.int32)
            if np.any((signed < -128)|(signed > 127)):
                raise ValueError('packed operand outside INT8')
            data=np.zeros((n+3)//4*4,dtype=np.uint8)
            data[:n]=signed.astype(np.int8).view(np.uint8)
            # Explicit shifts avoid depending on the host's byte order.
            b=data.reshape(-1,4).astype(np.uint32)
            out.append(b[:,0]|(b[:,1]<<8)|(b[:,2]<<16)|(b[:,3]<<24))
        else: out.append(values)
    if pos!=len(packet): raise ValueError('extra payload words')
    return np.concatenate(out)


def unpack_packet(packet):
    packet=np.asarray(packet,dtype=np.uint32)
    if len(packet)<16 or not packet[15]&1: raise ValueError('expected a packed packet')
    h=packet[:16].copy(); h[15]&=np.uint32(0xfffffffe)
    out=[h]; pos=16
    for is_byte,n in segments(h):
        count=(n+3)//4 if is_byte else n
        values=packet[pos:pos+count]; pos+=count
        if len(values)!=count: raise ValueError('short packed payload')
        if is_byte:
            b=np.column_stack([(values>>shift)&255 for shift in (0,8,16,24)]).astype(np.uint8).ravel()[:n]
            out.append(b.view(np.int8).astype(np.int32).view(np.uint32))
        else: out.append(values)
    if pos!=len(packet): raise ValueError('extra packed words')
    return np.concatenate(out)


def output_words(packet,logical_count):
    return (logical_count+3)//4 if int(packet[0]) in (0,1) else logical_count


def unpack_output(packet,data,logical_count):
    data=np.asarray(data,dtype=np.uint32)
    if len(data)!=output_words(packet,logical_count): raise ValueError('packed output length mismatch')
    if int(packet[0]) in (0,1):
        b=np.column_stack([(data>>shift)&255 for shift in (0,8,16,24)]).astype(np.uint8).ravel()[:logical_count]
        return b.view(np.int8).astype(np.int32)
    return data.view(np.int32).copy()
