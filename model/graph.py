"""
graph.py - the network as a list of layers, shared by the float model, the
integer model, the exporter and (later) the RS mapper and the hardware.

Each layer reads named tensors and writes one named tensor. Tensors are NumPy
arrays in N x C x H x W layout.

    conv : out = conv(in, W, b, stride, pad) [+ ReLU]
    add  : out = a + b [+ ReLU]                     (residual connection)
    gap  : out = sum over H x W                      (global pooling, as a sum)
    fc   : out = W @ in + b                          (the classifier)
"""

def conv(name, src, dst, cin, cout, k, stride, relu):
    return dict(op="conv", name=name, src=src, dst=dst, cin=cin, cout=cout,
                k=k, stride=stride, pad=k // 2, relu=relu)


def resnet8_graph():
    g = [conv("stem", "input", "a0", 3, 16, 3, 1, True)]
    t = "a0"
    for i, (cin, cout, stride) in enumerate([(16, 16, 1), (16, 32, 2), (32, 64, 2)], start=1):
        b = f"block{i}"
        g.append(conv(f"{b}.conv1", t, f"{b}.y1", cin, cout, 3, stride, True))
        g.append(conv(f"{b}.conv2", f"{b}.y1", f"{b}.y2", cout, cout, 3, 1, False))
        if stride != 1 or cin != cout:
            g.append(conv(f"{b}.shortcut", t, f"{b}.s", cin, cout, 1, stride, False))
            short = f"{b}.s"
        else:
            short = t
        g.append(dict(op="add", name=f"{b}.add", a=f"{b}.y2", b=short, dst=f"a{i}", relu=True))
        t = f"a{i}"
    g.append(dict(op="gap", name="pool", src=t, dst="pool"))
    g.append(dict(op="fc", name="fc", src="pool", dst="logits", cin=64, cout=10))
    return g
