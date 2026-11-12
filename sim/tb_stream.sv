`timescale 1ns/1ps
// Stimulus: 64-bit tokens (bit32=reference to previously emitted output).
// This feeds actual RTL tensors, rather than golden tensors, into later layers.
module tb_stream #(parameter integer COLS=8);
    reg clk=0; always #5 clk=~clk;
    reg resetn=0;
    reg [31:0] din;
    reg vin=0, ready=0;
    wire rin, vout, last, busy, error;
    wire [31:0] dout, cycles, packets;
    rs_accel_top #(.COLS(COLS)) dut(.aclk(clk),.aresetn(resetn),.s_axis_tdata(din),.s_axis_tkeep(4'hf),
        .s_axis_tvalid(vin),.s_axis_tready(rin),.s_axis_tlast(1'b0),
        .m_axis_tdata(dout),.m_axis_tvalid(vout),.m_axis_tready(ready),
        .m_axis_tlast(last),.m_axis_tkeep(),
        .s_axi_awaddr(8'd0),.s_axi_awvalid(1'b0),.s_axi_awready(),
        .s_axi_wdata(32'd0),.s_axi_wstrb(4'd0),.s_axi_wvalid(1'b0),.s_axi_wready(),
        .s_axi_bresp(),.s_axi_bvalid(),.s_axi_bready(1'b1),
        .s_axi_araddr(8'd0),.s_axi_arvalid(1'b0),.s_axi_arready(),
        .s_axi_rdata(),.s_axi_rresp(),.s_axi_rvalid(),.s_axi_rready(1'b1));
    assign busy=dut.core.busy || dut.ingress.m_valid || dut.ingress.count!=0;
    assign error=dut.core.error;
    assign cycles=dut.core.cycles;
    assign packets=dut.core.packets;
    integer stim, expf, logf, rc, ec, nout=0, nin=0, ticks=0;
    reg [63:0] token;
    reg [32:0] expected;
    reg [31:0] saved [0:200000];
    reg [31:0] lfsr=32'h1234beef;
    reg held=0;
    reg [32:0] held_word;
    reg done_input=0;
    integer mac_count[0:6*COLS-1],act_count[0:6*COLS-1],weight_count[0:6*COLS-1];
    integer emacs,eacts,eweights,metrics,mrc,total_macs,total_acts,total_weights,mi;
    genvar g,r,j;
    generate for(g=0;g<2;g=g+1) begin: mg
        for(r=0;r<3;r=r+1) begin: mr
            for(j=0;j<COLS;j=j+1) begin: mj
                localparam integer P=(g*3+r)*COLS+j;
                initial begin mac_count[P]=0; act_count[P]=0; weight_count[P]=0; end
                always @(posedge clk) if(resetn) begin
                    if(dut.core.groups[g].prows[r].pcols[j].pe.mac_en) mac_count[P]=mac_count[P]+1;
                    if(dut.core.groups[g].prows[r].pcols[j].pe.act_we) act_count[P]=act_count[P]+1;
                    if(dut.core.groups[g].prows[r].pcols[j].pe.weight_we) weight_count[P]=weight_count[P]+1;
                end
            end
        end
    end endgenerate
    string stim_name, exp_name;
    initial begin
        if(!$value$plusargs("STIM=%s",stim_name)) stim_name="stim.hex";
        if(!$value$plusargs("EXPECT=%s",exp_name)) exp_name="expect.hex";
        stim=$fopen(stim_name,"r"); expf=$fopen(exp_name,"r"); logf=$fopen("outputs.hex","w");
        if(!stim || !expf || !logf) $fatal(1,"cannot open vectors");
        metrics=$fopen("metrics.txt","r");
        mrc=$fscanf(metrics,"%d %d %d",emacs,eacts,eweights);
        if(mrc!=3) $fatal(1,"cannot read reuse metrics");
        repeat(4) @(negedge clk);
        resetn=1;
        while(!$feof(stim)) begin
            rc=$fscanf(stim,"%h\n",token);
            if(rc==1) begin
                repeat((nin%7)==0 ? 2 : 0) @(negedge clk);
                if(token[32]) begin
                    while(token[31:0]>=nout) @(negedge clk);
                    din=saved[token[31:0]];
                end else din=token[31:0];
                vin=1;
                @(posedge clk);
                while(!rin) @(posedge clk);
                @(negedge clk); vin=0; nin=nin+1;
            end
        end
        done_input=1;
    end
    always @(negedge clk) if(resetn) begin
        lfsr={lfsr[30:0],lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
        ready=lfsr[0] | lfsr[3];
    end
    always @(posedge clk) if(resetn) begin
        ticks=ticks+1;
        if(error) $fatal(1,"core rejected header input=%d",nin);
        if(held && (!vout || {last,dout}!==held_word)) $fatal(1,"AXIS output changed under stall");
        held=vout && !ready; held_word={last,dout};
        if(vout && ready) begin
            ec=$fscanf(expf,"%h\n",expected);
            if(ec!=1 || {last,dout}!==expected)
                $fatal(1,"output %d got %h last=%d expected %h",nout,dout,last,expected);
            saved[nout]=dout; nout=nout+1;
            $fwrite(logf,"%08h\n",dout);
        end
        if(done_input && !busy && !vout) begin
            ec=$fscanf(expf,"%h\n",expected);
            if(ec==1) $fatal(1,"missing output %d",nout);
            total_macs=0; total_acts=0; total_weights=0;
            for(mi=0;mi<6*COLS;mi=mi+1) begin
                total_macs=total_macs+mac_count[mi];
                total_acts=total_acts+act_count[mi];
                total_weights=total_weights+weight_count[mi];
            end
            if(total_macs!=emacs || total_acts!=eacts || total_weights!=eweights)
                $fatal(1,"reuse metrics mismatch MAC=%d/%d act=%d/%d weight=%d/%d",total_macs,emacs,total_acts,eacts,total_weights,eweights);
            $display("PASS REUSE macs=%0d PE_act_writes=%0d PE_weight_writes=%0d",total_macs,total_acts,total_weights);
            $display("PASS RTL inputs=%0d outputs=%0d packets=%0d busy_cycles=%0d wall_cycles=%0d",nin,nout,packets,cycles,ticks);
            $fclose(logf); $finish;
        end
        if(ticks>30000000) $fatal(1,"watchdog timeout");
    end
endmodule
