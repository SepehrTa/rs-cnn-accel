`timescale 1ns/1ps
module tb_resident #(
 parameter INPUTS=1,GOLDEN=1,COLS=14,FILTERS=4,P_CONTEXTS=2,Q_CONTEXTS=2
);
reg clk=0;always #5 clk=~clk;
reg rst=0;
reg [31:0] din=0;reg vin=0;wire rin;
wire [31:0] dout;wire vout,last;reg ready=0;
reg [31:0] stim[0:INPUTS-1],gold[0:GOLDEN-1];
integer tick=0,outputs=0,checks=0,layers=0,i,j,addr,base,count;
reg [31:0] actual;reg [7:0] octet;
rs_accel_top #(.COLS(COLS),.FILTERS(FILTERS),.P_CONTEXTS(P_CONTEXTS),.Q_CONTEXTS(Q_CONTEXTS)) dut(
 .aclk(clk),.aresetn(rst),
 .s_axi_awaddr(8'd0),.s_axi_awvalid(1'b0),.s_axi_awready(),
 .s_axi_wdata(32'd0),.s_axi_wstrb(4'd0),.s_axi_wvalid(1'b0),.s_axi_wready(),
 .s_axi_bresp(),.s_axi_bvalid(),.s_axi_bready(1'b1),
 .s_axi_araddr(8'd0),.s_axi_arvalid(1'b0),.s_axi_arready(),
 .s_axi_rdata(),.s_axi_rresp(),.s_axi_rvalid(),.s_axi_rready(1'b1),
 .s_axis_tdata(din),.s_axis_tkeep(4'hf),.s_axis_tvalid(vin),.s_axis_tready(rin),.s_axis_tlast(1'b0),
 .m_axis_tdata(dout),.m_axis_tkeep(),.m_axis_tvalid(vout),.m_axis_tready(ready),.m_axis_tlast(last));
always @(negedge clk) ready=(tick%7!=0 && tick%11!=0);
always @(posedge clk) if(rst) begin
 tick=tick+1;
 if(dut.resident.error || dut.core.error) $fatal(1,"hardware error resident=%d core=%d layer=%d position=%d",dut.resident.state,dut.core.state,dut.resident.layer,dut.resident.position);
 if(tick>20000000) $fatal(1,"timeout resident=%d core=%d layer=%d phase=%d ch=%d iy=%d ix=%d",dut.resident.state,dut.core.state,dut.resident.layer,dut.resident.payload_phase,dut.resident.ch,dut.resident.iy,dut.resident.ix);
 if(vout && ready) begin
  if(outputs==0) begin if(dout!==32'h52534e32 || !last) $fatal(1,"model upload ACK");end
  else begin
   if(dout!==gold[GOLDEN-10+(outputs-1)%10]) $fatal(1,"logit %d got %h expected %h",outputs-1,dout,gold[GOLDEN-10+(outputs-1)%10]);
   if(last!==(outputs%10==0)) $fatal(1,"result TLAST");
  end
  outputs=outputs+1;
 end
 if(dut.resident.state==27) begin
  base=dut.resident.d[3];count=dut.resident.d[13];
  for(j=0;j<count;j=j+1) begin
   if(dut.resident.d[14][0]) addr=(base>>2)+j;
   else addr=(base+j)>>2;
   actual={dut.resident.activation_banks[3].mem[addr],dut.resident.activation_banks[2].mem[addr],dut.resident.activation_banks[1].mem[addr],dut.resident.activation_banks[0].mem[addr]};
   if(!dut.resident.d[14][0]) begin octet=actual>>(((base+j)%4)*8);actual={{24{octet[7]}},octet};end
   if(actual!==gold[(checks%GOLDEN)+j]) $fatal(1,"layer=%d element=%d got=%h expected=%h",layers,j,actual,gold[(checks%GOLDEN)+j]);
  end
  checks=checks+count;layers=layers+1;
  $display("PASS LAYER %d elements=%d tick=%d",layers,count,tick);
 end
end
initial begin
 $readmemh("stim.hex",stim);$readmemh("expect.hex",gold);
 repeat(5) @(negedge clk);rst=1;
 for(i=0;i<INPUTS;i=i+1) begin
  @(negedge clk);din=stim[i];vin=1;
  @(posedge clk);while(!rin) @(posedge clk);
 end
 @(negedge clk);vin=0;
 wait(outputs==21);repeat(5) @(negedge clk);
 if(checks!=2*GOLDEN || layers!=28 || dut.core.macs!=25001984 || dut.resident.completed_images!=2) $fatal(1,"incomplete network/counters");
 $display("PASS RESIDENT exact_elements=%d layers=%d cycles=%d core_cycles=%d macs=%d model_reads=%d act_reads=%d act_write_bytes=%d overlap=%d wait=%d",checks,layers,dut.resident.network_cycles,dut.core.cycles,dut.core.macs,dut.resident.model_reads,dut.resident.activation_reads,dut.resident.activation_write_bytes,dut.core.overlap_cycles,dut.core.load_wait_cycles);
 $finish;
end
endmodule
