`timescale 1ns/1ps
module tb_control;
    reg clk=0; always #5 clk=~clk;
    reg rst=0;
    reg [7:0] awaddr=0,araddr=0;
    reg awvalid=0,wvalid=0,bready=0,arvalid=0,rready=0;
    reg [31:0] wdata=0;
    reg [3:0] wstrb=0;
    wire awready,wready,bvalid,arready,rvalid;
    wire [1:0] bresp,rresp;
    wire [31:0] rdata;
    reg [31:0] din=0; reg vin=0;
    wire rin;
    wire [31:0] dout;
    wire vout,last;
    integer result_count=0;
    rs_accel_top dut(.aclk(clk),.aresetn(rst),
        .s_axi_awaddr(awaddr),.s_axi_awvalid(awvalid),.s_axi_awready(awready),
        .s_axi_wdata(wdata),.s_axi_wstrb(wstrb),.s_axi_wvalid(wvalid),.s_axi_wready(wready),
        .s_axi_bresp(bresp),.s_axi_bvalid(bvalid),.s_axi_bready(bready),
        .s_axi_araddr(araddr),.s_axi_arvalid(arvalid),.s_axi_arready(arready),
        .s_axi_rdata(rdata),.s_axi_rresp(rresp),.s_axi_rvalid(rvalid),.s_axi_rready(rready),
        .s_axis_tdata(din),.s_axis_tkeep(4'hf),.s_axis_tvalid(vin),.s_axis_tready(rin),.s_axis_tlast(1'b0),
        .m_axis_tdata(dout),.m_axis_tkeep(),.m_axis_tvalid(vout),.m_axis_tready(1'b1),.m_axis_tlast(last));
    always @(posedge clk) if(rst && vout) begin
        if(dout!==32'hffffffff || !last) $fatal(1,"pool result after abort/reset is wrong");
        result_count=result_count+1;
    end
    task send_word(input [31:0] word);
        begin
            @(negedge clk); din=word; vin=1;
            @(posedge clk); while(!rin) @(posedge clk);
            @(negedge clk); vin=0;
        end
    endtask
    task send_aw(input [7:0] addr);
        begin
            @(negedge clk); awaddr=addr; awvalid=1;
            @(posedge clk); while(!awready) @(posedge clk);
            @(negedge clk); awvalid=0;
        end
    endtask
    task send_w(input [31:0] data,input [3:0] strb);
        begin
            @(negedge clk); wdata=data; wstrb=strb; wvalid=1;
            @(posedge clk); while(!wready) @(posedge clk);
            @(negedge clk); wvalid=0;
        end
    endtask
    task response(input [1:0] expected);
        begin
            @(negedge clk); while(!bvalid) @(negedge clk);
            repeat(4) begin
                @(negedge clk);
                if(!bvalid || bresp!==expected) $fatal(1,"write response unstable/wrong");
            end
            bready=1; @(negedge clk); bready=0;
        end
    endtask
    task read_reg(input [7:0] addr,input [31:0] expected,input [1:0] resp);
        begin
            @(negedge clk); araddr=addr; arvalid=1;
            @(posedge clk); while(!arready) @(posedge clk);
            @(negedge clk); arvalid=0;
            while(!rvalid) @(negedge clk);
            repeat(4) begin
                @(negedge clk);
                if(!rvalid || rdata!==expected || rresp!==resp) $fatal(1,"read unstable/wrong address %h got %h",addr,rdata);
            end
            rready=1; @(negedge clk); rready=0;
        end
    endtask
    integer i,wait_cycles;
    initial begin
        repeat(4) @(negedge clk); rst=1;
        read_reg(0,32'h52530003,0);
        read_reg(8,0,0);
        read_reg(8'h14,32'h02030820,0);
        read_reg(8'hfc,0,2);
        send_w(1,0); repeat(3) @(negedge clk); send_aw(4); response(0);
        send_aw(0); repeat(3) @(negedge clk); send_w(1,15); response(2);
        // Invalid opcode enters sticky fault. Reset must flush FIFO and recover.
        for(i=0;i<16;i=i+1) begin
            @(negedge clk); din=(i==0 ? 99 : 0); vin=1;
            @(posedge clk); while(!rin) @(posedge clk);
            @(negedge clk); vin=0;
        end
        repeat(20) @(negedge clk);
        read_reg(8,3,0);
        send_aw(4); repeat(3) @(negedge clk); send_w(1,15); response(0);
        read_reg(8,0,0); read_reg(12,0,0); read_reg(16,0,0);
        // Abort an incomplete valid pooling command, then execute a fresh one.
        for(i=0;i<16;i=i+1) send_word(i==0 ? 2 : i==1 ? 64 : i==2 ? 1 : 0);
        repeat(5) send_word(13);
        read_reg(8,1,0);
        send_w(1,15); send_aw(4); response(0);
        read_reg(8,0,0);
        for(i=0;i<16;i=i+1) send_word(i==0 ? 2 : i==1 ? 2 : i==2 ? 1 : 0);
        send_word(-128); send_word(127);
        for(wait_cycles=0;wait_cycles<256 && result_count!=1;wait_cycles=wait_cycles+1) @(negedge clk);
        if(result_count!=1) $fatal(1,"missing output after abort/reset");
        read_reg(16,1,0);
        // Reset with address/operand/product commits in flight, then prove a
        // new convolution clears stale psums and cannot emit an old product.
        fork
            begin
                for(i=0;i<16;i=i+1) send_word(i==0 ? 0 : i>=1 && i<=6 ? 1 : i==7 ? 64 : i==10 ? 1073741824 : i==11 ? 30 : 0);
                repeat(128) send_word(1);
            end
            begin wait(dut.core.state==4);end
        join_any
        disable fork;
        @(negedge clk);vin=0;
        send_w(1,15);send_aw(4);response(0);
        for(i=0;i<16;i=i+1) send_word(i==0 ? 0 : i>=1 && i<=7 ? 1 : i==10 ? 1073741824 : i==11 ? 30 : 0);
        send_word(-1);send_word(1);
        for(wait_cycles=0;wait_cycles<256 && result_count!=2;wait_cycles=wait_cycles+1) @(negedge clk);
        if(result_count!=2) $fatal(1,"missing convolution after MAC pipeline abort");
        read_reg(16,1,0);
        $display("PASS CONTROL independent AW/W, stalls, decode, fault, pooling/MAC abort and reset recovery");
        $finish;
    end
    initial begin #100000; $fatal(1,"control watchdog"); end
endmodule
