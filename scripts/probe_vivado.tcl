set root [file normalize [file join [file dirname [info script]] ..]]
file mkdir [file join $root build]
set f [open [file join $root build vivado_environment.txt] w]
puts $f [version]
puts $f "PYNQ boards: [get_board_parts -quiet *pynq*]"
puts $f "PS7: [get_ipdefs -quiet xilinx.com:ip:processing_system7:*]"
close $f
