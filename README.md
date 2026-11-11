# RS-CNN Accelerator

An Eyeriss-style CNN accelerator for the PYNQ-Z2 (Zynq-7020), built around the
Row Stationary dataflow from Chen et al., *"Eyeriss: An Energy-Efficient
Reconfigurable Accelerator for Deep Convolutional Neural Networks"* (JSSC 2017).
The target network is ResNet-8 on CIFAR-10, the MLPerf Tiny image-classification model,
in int8.

## The model

`model/` trains ResNet-8 in PyTorch and turns it into the integer model the
hardware will run. The integer model in `model/nn_numpy.py` is the golden
reference: plain NumPy, int8 activations and weights, 32-bit accumulation, and
fixed-point requantization with a single rounding step, so the hardware can be
checked against it bit for bit.

```sh
pip install -r requirements.txt
cd model
python train.py            # trains on CIFAR-10 (downloaded to model/data), writes resnet8_float.npz
python quantize.py         # folds batch norm, calibrates, quantizes, writes resnet8_int8.npz
```

`quantize.py` prints the float and integer accuracy on the full test set and
exports the integer parameters, the layer graph, and every intermediate tensor
for a few test images, for checking the hardware layer by layer.

`tests/test_nn_numpy.py` checks the NumPy models without any trained weights:
the convolution against a direct loop implementation, batch-norm folding,
requantization against exact integer arithmetic, and the integer model against
the float model.

## Repo layout

```
model/     ResNet-8 (PyTorch), training, quantization, float and integer NumPy models
host/      Python for the PYNQ board
hw/        RTL
sim/       testbenches
tests/     tests of the Python models
docs/      design notes
```

---

Sepehr Tabrizchi, [github.com/SepehrTa](https://github.com/SepehrTa)
