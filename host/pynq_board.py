"""
pynq_board.py - load the accelerator's bitstream on the PYNQ-Z2 and reach its blocks.

Reused from the riscv-ml-core project (riscv_loader.py), generalized: the
bitstream is loaded by PYNQ together with its .hwh (same base name), the fabric
clock is then set explicitly (the dividers in the .hwh assume PS settings that
may differ from how the board booted), and any AXI block can be found by name.

    from pynq_board import Board
    board = Board("rs_accel.bit", fclk_mhz=100)
    regs = board.mmio("accel_ctrl")        # an AXI-Lite block, by its block-design name
"""
from pynq import MMIO, Clocks, Overlay


class Board:
    def __init__(self, bitfile, fclk_mhz=100.0):
        self.overlay = Overlay(bitfile)
        Clocks.fclk0_mhz = fclk_mhz
        print(f"loaded {bitfile}, FCLK0 = {Clocks.fclk0_mhz:.1f} MHz")

    def find(self, name):
        """(base address, size) of a block, from the overlay's ip_dict or mem_dict."""
        for table in (self.overlay.ip_dict, self.overlay.mem_dict):
            for key, info in table.items():
                if key == name or key.endswith("/" + name):
                    base = info.get("phys_addr", info.get("base_address"))
                    return int(base), int(info["addr_range"])
        raise KeyError(f"{name} not found in the overlay; check the .hwh file")

    def mmio(self, name):
        base, size = self.find(name)
        return MMIO(base, size)
