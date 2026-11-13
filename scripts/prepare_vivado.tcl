# Open and prepare the committed project after a fresh clone. No board or
# Python required. Run with Vivado 2025.2 or Tools > Run Tcl Script in its GUI.
set project_root [file normalize [file join [file dirname [info script]] ..]]
set_param general.maxThreads 4
set project_file [file join $project_root rs-cnn-accel.xpr]
if {[current_project -quiet] eq ""} {
    open_project $project_file
} elseif {[file normalize [get_property DIRECTORY [current_project]]] ne $project_root} {
    error "A different project is open. Close it before preparing $project_file"
}

# Module-reference tops require Verilog in 2025.2; children remain SV.
set_property FILE_TYPE Verilog [get_files {rs_accel_top.sv rs_ddr_monitor.sv}]
set_property TOP system_wrapper [get_filesets sources_1]
set_property TOP_AUTO_SET 0 [get_filesets sources_1]
# A source clone has no incremental synthesis checkpoint to reuse.
set_property AUTO_INCREMENTAL_CHECKPOINT 0 [get_runs synth_1]
set_property INCREMENTAL_CHECKPOINT {} [get_runs synth_1]
# Apply post-route physical optimization to the 100 MHz implementation.
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]

foreach project_input [get_files -norecurse] {
    if {![file exists $project_input]} {error "Missing project input: $project_input"}
}
set bd_file [get_files system.bd]
if {[llength $bd_file] != 1} {error "Expected one system.bd block design"}
open_bd_design $bd_file
foreach module_ip {system_rs_accel_0_0 system_rs_ddr_monitor_0_0} {
    update_module_reference [get_ips $module_ip]
}
validate_bd_design
save_bd_design
generate_target all $bd_file
update_compile_order -fileset sources_1
puts "READY: rs-cnn-accel; top=system_wrapper; part=[get_property PART [current_project]]"
puts "Next: Generate Bitstream, or open block design system in the GUI."
