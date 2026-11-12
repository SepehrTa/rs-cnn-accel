# Usage: vivado -mode batch -source scripts/synth.tcl
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file join $root build synth]
file mkdir $out
cd $out
create_project -in_memory -part xc7z020clg400-1
set_param general.maxThreads 4
read_verilog -sv [glob [file join $root hw *.sv]]
read_xdc [file join $root hw fclk.xdc]
set cols 8
if {[llength $argv]>0} {set cols [lindex $argv 0]}
synth_design -top rs_accel_top -part xc7z020clg400-1 -mode out_of_context -generic COLS=$cols
report_utilization -file [file join $out utilization_synth.rpt]
report_timing_summary -file [file join $out timing_synth.rpt]
report_drc -file [file join $out drc_synth.rpt]
write_checkpoint -force [file join $out rs_accel_synth.dcp]
# Out-of-context placement/routing checks the accelerator fabric. Board pin DRC
# is intentionally separate; this is an internal PS peripheral, not top-level IO.
opt_design
place_design
phys_opt_design
route_design
report_utilization -file [file join $out utilization_route.rpt]
report_timing_summary -file [file join $out timing_route.rpt]
report_drc -file [file join $out drc_route.rpt]
write_checkpoint -force [file join $out rs_accel_route.dcp]
