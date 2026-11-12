`timescale 1ns/1ps
module tb_pe;
    reg clk=0; always #5 clk=~clk;
    reg awe=0,wwe=0,sample=0,mac=0,first=0;
    reg [7:0] aa=0,sa=0,oa=0,ra=0;
    reg [1:0] wa=0,tap=0;
    reg signed [7:0] ad=0,wd=0;
    wire signed [31:0] result;
    reg reduce=0;
    wire signed [31:0] link;
    rs_pe dut(.clk(clk),.act_we(awe),.act_addr(aa),.act_data(ad),
        .weight_we(wwe),.weight_addr(wa),.weight_data(wd),.sample_en(sample),
        .mac_en(mac),.first_mac(first),.sample_addr(sa),.tap(tap),
        .out_addr(oa),.read_addr(ra),.psum(result),.reduce_en(reduce),
        .neighbor_psum(-32'sd123),.link_psum(link));
    integer expected[0:31];
    integer channel,x,s,i,a,w;
    reg [31:0] rng=32'hbeef1234;
    reg signed [7:0] acts[0:33],weights[0:2];
    initial begin
        for(i=0;i<32;i=i+1) expected[i]=0;
        for(channel=0;channel<17;channel=channel+1) begin
            for(i=0;i<34;i=i+1) begin
                rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
                acts[i]=(channel==0 ? -128 : rng[7:0]);
                @(negedge clk); aa=i; ad=acts[i]; awe=1;
            end
            @(negedge clk); awe=0;
            for(i=0;i<3;i=i+1) begin
                rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
                weights[i]=(channel==0 ? -128 : rng[7:0]);
                @(negedge clk); wa=i; wd=weights[i]; wwe=1;
            end
            @(negedge clk); wwe=0;
            for(x=0;x<32;x=x+1) begin
                for(s=0;s<3;s=s+1) begin
                    @(negedge clk); sa=x+s; tap=s; oa=x; ra=x; sample=1;
                    @(negedge clk); sample=0; mac=1; first=(channel==0 && s==0);
                    a=acts[x+s]; w=weights[s]; expected[x]=expected[x]+a*w;
                    @(negedge clk); mac=0; first=0;
                    if(result!==expected[x]) $fatal(1,"PE ch=%d x=%d tap=%d got=%d expected=%d",channel,x,s,result,expected[x]);
                end
                @(negedge clk); reduce=1;
                @(negedge clk); reduce=0;
                if(link!==expected[x]-123) $fatal(1,"PE neighbor accumulation wrong");
            end
        end
        $display("PASS PE 1632 signed MACs, 32 psum addresses, 17 channel loads, 544 neighbor reductions");
        $finish;
    end
endmodule
