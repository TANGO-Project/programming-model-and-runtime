open_project matrixmul_zynq_prj
set_top matrixmul
add_files      src/matrixmul_zynq.cpp 
add_files -tb  src/block_mult_vivado.cpp 
open_solution solution1 -reset
set_part xc7z020clg484-1  
create_clock -period "100MHz"
exit
