# Internal accelerator OOC clock; board design uses PS7 FCLK_CLK0, not a PL pin.
create_clock -name fclk0 -period 10.000 [get_ports aclk]
set_input_delay -clock fclk0 2.000 [get_ports -filter {DIRECTION == IN && NAME != aclk}]
set_output_delay -clock fclk0 2.000 [get_ports -filter {DIRECTION == OUT}]
