`timescale 1ns/1ps
// Stimulus: 64-bit tokens (bit32=reference to previously emitted output).
// This feeds actual RTL tensors, rather than golden tensors, into later layers.
module tb_stream #(parameter integer COLS=8, PACKED=0, FILTERS=2, P_CONTEXTS=1, Q_CONTEXTS=1);
    reg clk=0; always #5 clk=~clk;
    reg resetn=0;
    reg [31:0] din;
    reg vin=0, ready=0;
    wire rin, vout, last, busy, error;
    wire [31:0] dout, cycles, packets;
    rs_accel_top #(.COLS(COLS),.FILTERS(FILTERS),.P_CONTEXTS(P_CONTEXTS),.Q_CONTEXTS(Q_CONTEXTS)) dut(.aclk(clk),.aresetn(resetn),.s_axis_tdata(din),.s_axis_tkeep(4'hf),
        .s_axis_tvalid(vin),.s_axis_tready(rin),.s_axis_tlast(1'b0),
        .m_axis_tdata(dout),.m_axis_tvalid(vout),.m_axis_tready(ready),
        .m_axis_tlast(last),.m_axis_tkeep(),
        .s_axi_awaddr(8'd0),.s_axi_awvalid(1'b0),.s_axi_awready(),
        .s_axi_wdata(32'd0),.s_axi_wstrb(4'd0),.s_axi_wvalid(1'b0),.s_axi_wready(),
        .s_axi_bresp(),.s_axi_bvalid(),.s_axi_bready(1'b1),
        .s_axi_araddr(8'd0),.s_axi_arvalid(1'b0),.s_axi_arready(),
        .s_axi_rdata(),.s_axi_rresp(),.s_axi_rvalid(),.s_axi_rready(1'b1));
    assign busy=dut.core.busy || dut.codec.busy || dut.resident_busy || dut.ingress.m_valid || dut.ingress.count!=0;
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
    reg previous_mac=0;
    integer header[0:15], opcode[0:9999];
    integer in_packet=0,out_packet=0,semantic_in=0,physical_out=0,hh,fc,nbytes,bb;
    reg [31:0] scalar,packed_word,actual_word;
    reg expected_last;
    task read_scalar(output reg [31:0] value);
        begin
            rc=$fscanf(stim,"%h\n",token);
            if(rc!=1) $fatal(1,"truncated semantic packet");
            if(token[32]) begin
                while(token[31:0]>=nout) @(negedge clk);
                value=saved[token[31:0]];
            end else value=token[31:0];
            semantic_in=semantic_in+1;
        end
    endtask
    task send_word(input reg [31:0] value);
        begin
            repeat((nin%7)==0 ? 2 : 0) @(negedge clk);
            din=value; vin=1;
            @(posedge clk);
            while(!rin) @(posedge clk);
            @(negedge clk); vin=0; nin=nin+1;
        end
    endtask
    task send_segment(input integer count,input integer bytes);
        integer si,used;
        reg [31:0] value,word_value;
        begin
            used=0; word_value=0;
            for(si=0;si<count;si=si+1) begin
                read_scalar(value);
                if(PACKED && bytes) begin
                    if($signed(value)<-128 || $signed(value)>127) $fatal(1,"non-INT8 byte operand");
                    word_value=word_value | ((value & 255) << (8*used));
                    if(used==3 || si==count-1) begin send_word(word_value); used=0; word_value=0; end
                    else used=used+1;
                end else send_word(value);
            end
        end
    endtask
    integer mac_count[0:FILTERS*3*COLS-1],act_count[0:FILTERS*3*COLS-1],weight_count[0:FILTERS*3*COLS-1];
    integer emacs,eacts,eweights,metrics,mrc,total_macs,total_acts,total_weights,mi;
    genvar g,r,j;
    generate for(g=0;g<FILTERS;g=g+1) begin: mg
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
                if(token[32]) $fatal(1,"header cannot be a dependency token");
                header[0]=token[31:0]; opcode[in_packet]=token[31:0];
                in_packet=in_packet+1; semantic_in=semantic_in+1;
                send_word(token[31:0]);
                for(hh=1;hh<16;hh=hh+1) begin
                    read_scalar(scalar); header[hh]=scalar;
                    send_word(hh==15 && PACKED ? scalar|1 : scalar);
                end
                case(header[0])
                0: begin
                    if(header[6]>2) send_segment(3*(header[6]-2),0);
                    send_segment(header[7]*(((header[5]-1)*header[2]+header[1])*header[3]+header[6]*header[1]*header[1]),1);
                end
                1: send_segment(2*header[1],1);
                2: send_segment(header[1]*header[2],1);
                3: begin
                    send_segment(header[1],0);
                    for(fc=0;fc<header[2];fc=fc+1) begin send_segment(1,0); send_segment(header[1],1); end
                end
                default: $fatal(1,"bad stimulus opcode");
                endcase
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
        if(previous_mac && dut.core.state==dut.core.MAC_FETCH) $fatal(1,"MAC initiation interval regressed to two clocks");
        previous_mac=dut.core.state==dut.core.MAC;
        if(dut.core.state==dut.core.MAC && dut.core.input_fire &&
           (dut.core.ld_state==dut.core.LD_ACT || dut.core.ld_state==dut.core.LD_WEIGHT) &&
           dut.core.ld_bank==dut.core.compute_bank) $fatal(1,"loader overwrote active compute bank");
        if(error) $fatal(1,"core rejected header input=%d",nin);
        if(held && (!vout || {last,dout}!==held_word)) $fatal(1,"AXIS output changed under stall");
        held=vout && !ready; held_word={last,dout};
        if(vout && ready) begin
            physical_out=physical_out+1; expected_last=0;
            if(PACKED && (opcode[out_packet]==0 || opcode[out_packet]==1)) begin
                for(bb=0;bb<4 && !expected_last;bb=bb+1) begin
                    actual_word={{24{dout[8*bb+7]}},dout[8*bb+:8]};
                    ec=$fscanf(expf,"%h\n",expected);
                    if(ec!=1 || actual_word!==expected[31:0])
                        $fatal(1,"packed output %d got %h expected %h",nout,actual_word,expected);
                    saved[nout]=actual_word; nout=nout+1; expected_last=expected[32];
                    $fwrite(logf,"%08h\n",actual_word);
                    if(expected_last && bb<3 && (dout >> (8*(bb+1)))!=0) $fatal(1,"nonzero output tail padding");
                end
                if(last!==expected_last) $fatal(1,"packed TLAST mismatch");
            end else begin
                ec=$fscanf(expf,"%h\n",expected);
                if(ec!=1 || {last,dout}!==expected)
                    $fatal(1,"output %d got %h last=%d expected %h",nout,dout,last,expected);
                saved[nout]=dout; nout=nout+1;
                $fwrite(logf,"%08h\n",dout);
            end
            if(last) out_packet=out_packet+1;
        end
        if(done_input && !busy && !vout) begin
            if(dut.codec.input_words!=nin || dut.codec.output_words!=physical_out)
                $fatal(1,"physical transport counters differ from actual handshakes");
            ec=$fscanf(expf,"%h\n",expected);
            if(ec==1) $fatal(1,"missing output %d",nout);
            total_macs=0; total_acts=0; total_weights=0;
            for(mi=0;mi<FILTERS*3*COLS;mi=mi+1) begin
                total_macs=total_macs+mac_count[mi];
                total_acts=total_acts+act_count[mi];
                total_weights=total_weights+weight_count[mi];
            end
            if(total_macs!=emacs || total_acts!=eacts || total_weights!=eweights)
                $fatal(1,"reuse metrics mismatch MAC=%d/%d act=%d/%d weight=%d/%d",total_macs,emacs,total_acts,eacts,total_weights,eweights);
            if(dut.core.macs!=total_macs) $fatal(1,"hardware MAC counter differs from PE enables");
            $display("PASS REUSE macs=%0d PE_act_writes=%0d PE_weight_writes=%0d",total_macs,total_acts,total_weights);
            $display("PASS RTL inputs=%0d outputs=%0d packets=%0d busy_cycles=%0d wall_cycles=%0d",nin,nout,packets,cycles,ticks);
            $display("PASS TRANSPORT packed=%0d semantic_inputs=%0d physical_inputs=%0d physical_outputs=%0d",PACKED,semantic_in,nin,physical_out);
            $display("PASS CONTEXTS p=%0d q=%0d overlap_cycles=%0d load_wait_cycles=%0d",P_CONTEXTS,Q_CONTEXTS,dut.core.overlap_cycles,dut.core.load_wait_cycles);
            $fclose(logf); $finish;
        end
        if(ticks>30000000) $fatal(1,"watchdog timeout");
    end
endmodule
