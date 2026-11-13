`timescale 1ns/1ps
module tb_ddr_monitor;
reg clk=0;always #5 clk=~clk;
reg rst=0;reg [7:0] aw=0,ar=0;reg av=0,wv=0,rv=0;reg [31:0] wd=0;
wire aready,wready,bvalid,arready,rvalid;wire [31:0] rd;
reg r_valid=0,r_ready=0,w_valid=0,w_ready=0,ar_valid=0,ar_ready=0,aw_valid=0,aw_ready=0;
reg [7:0] strb=0;
rs_ddr_monitor dut(.aclk(clk),.aresetn(rst),.s_axi_awaddr(aw),.s_axi_awvalid(av),.s_axi_awready(aready),
 .s_axi_wdata(wd),.s_axi_wstrb(4'hf),.s_axi_wvalid(wv),.s_axi_wready(wready),.s_axi_bresp(),.s_axi_bvalid(bvalid),.s_axi_bready(1'b1),
 .s_axi_araddr(ar),.s_axi_arvalid(rv),.s_axi_arready(arready),.s_axi_rdata(rd),.s_axi_rresp(),.s_axi_rvalid(rvalid),.s_axi_rready(1'b1),
 .hp_rvalid(r_valid),.hp_rready(r_ready),.hp_wvalid(w_valid),.hp_wready(w_ready),.hp_wstrb(strb),
 .hp_arvalid(ar_valid),.hp_arready(ar_ready),.hp_awvalid(aw_valid),.hp_awready(aw_ready));
task check(input [7:0] address,input [31:0] expected);
begin
 @(negedge clk);ar=address;rv=1;
 @(posedge clk);while(!arready) @(posedge clk);
 @(negedge clk);rv=0;
 if(!rvalid || rd!==expected) $fatal(1,"monitor address=%h got=%d expected=%d",address,rd,expected);
end endtask
initial begin
 repeat(4) @(negedge clk);rst=1;
 @(negedge clk);r_valid=1;w_valid=1;ar_valid=1;aw_valid=1;strb=8'h95;
 repeat(3) @(negedge clk); // stalled cycles must not count
 r_ready=1;w_ready=1;ar_ready=1;aw_ready=1;
 repeat(5) @(negedge clk);
 r_valid=0;w_valid=0;ar_valid=0;aw_valid=0;
 check(0,32'h5253444d);check(8,5);check(12,5);check(16,20);check(20,5);check(24,5);
 // Independent W then AW arrival.
 @(negedge clk);wd=1;wv=1;
 @(posedge clk);while(!wready) @(posedge clk);
 @(negedge clk);wv=0;
 repeat(3) @(negedge clk);aw=4;av=1;
 @(posedge clk);while(!aready) @(posedge clk);
 @(negedge clk);av=0;
 repeat(5) @(negedge clk);
 check(8,0);check(12,0);check(16,0);
 $display("PASS DDR MONITOR handshakes, stalls, byte strobes, independent AW/W and reset");$finish;
end
endmodule
