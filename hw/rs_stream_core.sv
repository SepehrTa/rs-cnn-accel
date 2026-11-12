// Packet-driven RS engine. All state advances only on accepted words or compute
// clocks. AXIS output is stable under backpressure. See docs/ARCHITECTURE.md.
module rs_stream_core #(
    parameter integer FILTERS = 2,
    parameter integer COLS = 8,
    parameter integer KMAX = 3,
    parameter integer MAX_WIDTH = 34,
    parameter integer MAX_OUT = 32,
    parameter integer MAX_FC = 256
) (
    input wire aclk, input wire aresetn,
    input wire [31:0] s_axis_tdata, input wire s_axis_tvalid,
    output wire s_axis_tready, input wire s_axis_tlast,
    output reg [31:0] m_axis_tdata, output wire m_axis_tvalid,
    input wire m_axis_tready, output wire m_axis_tlast,
    output wire busy, output reg error,
    output reg [31:0] cycles, output reg [31:0] packets
);
    localparam HEADER=0, CHECK=1, ACT=2, WEIGHT=3, MAC=4,
               REDUCE=5, QUANT=6, EMIT=7, ADD_A=8, ADD_B=9,
               ADD_CALC=10, POOL=11, FC_INPUT=12, FC_BIAS=13,
               FC_WEIGHT=14, FAULT=15, POST_START=16, POST_WAIT=17,
               MAC_FETCH=18, ADD_SUM=19, ADD_SHIFT=20, FC_COMB=21, FC_ACC=22;
    reg [4:0] state;
    reg [31:0] h [0:15];
    reg [3:0] hi;
    reg [7:0] iy, ix, wg, wr, wt, ox, tap, og, oj, rr;
    reg [31:0] channel, index, group_index;
    reg signed [31:0] acc, operand_a;
    reg signed [31:0] fc_input [0:MAX_FC-1];
    wire [7:0] k = h[1][7:0], stride = h[2][7:0];
    wire [7:0] width = h[3][7:0], out_width = h[4][7:0];
    wire [7:0] rows = h[5][7:0], filters = h[6][7:0];
    wire [7:0] patch_rows = (stride==2 ? ((rows-1)<<1) : rows-1)+k;
    // Runtime stride has two legal values. A generic 32x32 validation multiply
    // here used to dominate timing despite never participating in inference.
    wire [8:0] required_width = (stride==2 ? {out_width,1'b0}-9'd2 : {1'b0,out_width}-9'd1)+{1'b0,k};
    wire input_fire = s_axis_tvalid && s_axis_tready;
    wire output_fire = m_axis_tvalid && m_axis_tready;
    wire last_conv = og==filters-1 && oj==rows-1 && ox==out_width-1;
    wire final_word = h[0]==0 ? last_conv :
                      h[0]==1 ? index==h[1]-1 : group_index==h[2]-1;
    assign s_axis_tready = aresetn && (state==HEADER || state==ACT || state==WEIGHT ||
                           state==ADD_A || state==ADD_B || state==POOL ||
                           state==FC_INPUT || state==FC_BIAS || state==FC_WEIGHT);
    assign m_axis_tvalid = aresetn && state==EMIT;
    assign m_axis_tlast = m_axis_tvalid && final_word;
    assign busy = state!=HEADER || hi!=0;

    wire signed [31:0] pe_sums [0:FILTERS*KMAX*COLS-1];
    wire signed [31:0] pe_links [0:FILTERS*KMAX*COLS-1];
    genvar g,r,j;
    generate for(g=0;g<FILTERS;g=g+1) begin: groups
        for(r=0;r<KMAX;r=r+1) begin: prows
            for(j=0;j<COLS;j=j+1) begin: pcols
                localparam integer P=(g*KMAX+r)*COLS+j;
                wire signed [31:0] neighbor_psum;
                if(r==0) assign neighbor_psum=32'sd0;
                else assign neighbor_psum=pe_links[(g*KMAX+r-1)*COLS+j];
                // Diagonal multicast: identical source row is written to all
                // row pairs requiring iy=j*stride+r, and both filter groups.
                rs_pe #(.MAX_WIDTH(MAX_WIDTH),.MAX_OUT(MAX_OUT),.MAX_KERNEL(KMAX)) pe (
                    .clk(aclk),
                    .act_we(input_fire && state==ACT && j<rows && r<k && g<filters && iy==j*stride+r),
                    .act_addr(ix), .act_data(s_axis_tdata[7:0]),
                    .weight_we(input_fire && state==WEIGHT && g==wg && r==wr && j<rows),
                    .weight_addr(wt[1:0]), .weight_data(s_axis_tdata[7:0]),
                    .sample_en(state==MAC_FETCH && g<filters && r<k && j<rows),
                    .mac_en(state==MAC && g<filters && r<k && j<rows),
                    .first_mac(channel==0 && tap==0),
                    .sample_addr((stride==2 ? (ox<<1) : ox)+tap), .tap(tap[1:0]),
                    .out_addr(ox), .read_addr(ox), .psum(pe_sums[P]),
                    .reduce_en(state==REDUCE && g==og && j==oj && r==rr),
                    .neighbor_psum(neighbor_psum),.link_psum(pe_links[P])
                );
            end
        end
    end endgenerate

    wire signed [31:0] multiplier = og==0 ? h[10] : h[13];
    wire [5:0] shift = og==0 ? h[11][5:0] : h[14][5:0];
    wire signed [31:0] bias = og==0 ? h[9] : h[12];
    wire signed [31:0] quant_result;
    wire quant_valid;
    rs_postprocess post(.clk(aclk),.resetn(aresetn),.valid_in(state==POST_START),
                        .value(acc),.multiplier(multiplier),.shift(shift),
                        .relu(h[8][0]),.valid_out(quant_valid),.result(quant_result));
    wire signed [63:0] a_scaled = $signed(operand_a[7:0])*$signed(h[10]);
    wire signed [63:0] b_scaled = $signed(s_axis_tdata[7:0])*$signed(h[13]);
    reg signed [63:0] add_sum, add_pa, add_pb, add_rounded;
    (* use_dsp="yes" *) reg signed [24:0] fc_lo;
    (* use_dsp="yes" *) reg signed [23:0] fc_hi;
    reg signed [31:0] fc_product;
    integer z;
    always @(posedge aclk) begin
        if (!aresetn) begin
            state<=HEADER; hi<=0; error<=0; cycles<=0; packets<=0;
            m_axis_tdata<=0; channel<=0; index<=0; group_index<=0;
            iy<=0; ix<=0; wg<=0; wr<=0; wt<=0; ox<=0; tap<=0;
            og<=0; oj<=0; rr<=0; acc<=0; operand_a<=0; add_sum<=0;
            for(z=0;z<16;z=z+1) h[z]<=0;
        end else begin
            if (busy) cycles<=cycles+1;
            case(state)
            HEADER: if(input_fire) begin
                h[hi]<=s_axis_tdata;
                if(hi==15) begin hi<=0; state<=CHECK; end
                else hi<=hi+1;
            end
            CHECK: begin
                channel<=0; index<=0; group_index<=0; iy<=0; ix<=0;
                wg<=0; wr<=0; wt<=0; ox<=0; tap<=0; og<=0; oj<=0; acc<=0;
                if(h[0]==0 && (h[1]==1 || h[1]==3) && h[1]<=KMAX &&
                   (h[2]==1 || h[2]==2) && h[3]>=h[1] && h[3]<=MAX_WIDTH &&
                   h[4]>0 && h[4]<=MAX_OUT && required_width<={1'b0,width} &&
                   h[5]>0 && h[5]<=COLS && h[6]>0 && h[6]<=FILTERS &&
                   h[7]>0 && h[7]<=65535 && h[10]>0 && h[10]<32'h80000000 &&
                   h[11]>=1 && h[11]<=62 && (h[6]==1 ||
                   (h[13]>0 && h[13]<32'h80000000 && h[14]>=1 && h[14]<=62))) state<=ACT;
                else if(h[0]==1 && h[1]>0 && h[10]<32'h80000000 &&
                        h[13]<32'h80000000 && h[11]>=1 && h[11]<=62) state<=ADD_A;
                else if(h[0]==2 && h[1]>0 && h[1]<=65535 && h[2]>0 && h[2]<=65535) state<=POOL;
                else if(h[0]==3 && h[1]>0 && h[1]<=MAX_FC && h[2]>0 && h[2]<=65535) state<=FC_INPUT;
                else begin state<=FAULT; error<=1; end
            end
            ACT: if(input_fire) begin
                if(ix==width-1) begin
                    ix<=0;
                    if(iy==patch_rows-1) begin iy<=0; state<=WEIGHT; end
                    else iy<=iy+1;
                end else ix<=ix+1;
            end
            WEIGHT: if(input_fire) begin
                if(wt==k-1) begin
                    wt<=0;
                    if(wr==k-1) begin
                        wr<=0;
                        if(wg==filters-1) begin wg<=0; ox<=0; tap<=0; state<=MAC_FETCH; end
                        else wg<=wg+1;
                    end else wr<=wr+1;
                end else wt<=wt+1;
            end
            MAC_FETCH: state<=MAC;
            MAC: begin
                state<=MAC_FETCH;
                if(tap==k-1) begin
                    tap<=0;
                    if(ox==out_width-1) begin
                        ox<=0;
                        if(channel==h[7]-1) begin rr<=0; state<=REDUCE; end
                        else begin channel<=channel+1; state<=ACT; end
                    end else ox<=ox+1;
                end else tap<=tap+1;
            end
            REDUCE: begin
                // Enable one directly wired neighbor hop per clock. Arithmetic
                // and link storage are inside each PE, never a shared reducer.
                if(rr==k-1) state<=QUANT;
                else rr<=rr+1;
            end
            QUANT: begin acc<=pe_links[(og*KMAX+k-1)*COLS+oj]+bias; state<=POST_START; end
            POST_START: state<=POST_WAIT;
            POST_WAIT: if(quant_valid) begin m_axis_tdata<=quant_result; state<=EMIT; end
            ADD_CALC: begin
                m_axis_tdata<=add_rounded>127 ? 32'd127 :
                     add_rounded<(h[8][0] ? 0 : -128) ? (h[8][0] ? 32'd0 : -32'sd128) : add_rounded[31:0];
                state<=EMIT;
            end
            ADD_A: if(input_fire) begin operand_a<=$signed(s_axis_tdata); state<=ADD_B; end
            ADD_B: if(input_fire) begin add_pa<=a_scaled; add_pb<=b_scaled; state<=ADD_SUM; end
            ADD_SUM: begin add_sum<=add_pa+add_pb+(64'sd1 <<< (h[11][5:0]-1)); state<=ADD_SHIFT; end
            ADD_SHIFT: begin add_rounded<=add_sum >>> h[11][5:0]; state<=ADD_CALC; end
            POOL: if(input_fire) begin
                if(index==h[1]-1) begin
                    m_axis_tdata<=acc+$signed(s_axis_tdata); acc<=0; index<=0; state<=EMIT;
                end else begin acc<=acc+$signed(s_axis_tdata); index<=index+1; end
            end
            FC_INPUT: if(input_fire) begin
                fc_input[index]<=$signed(s_axis_tdata);
                if(index==h[1]-1) begin index<=0; state<=FC_BIAS; end
                else index<=index+1;
            end
            FC_BIAS: if(input_fire) begin acc<=$signed(s_axis_tdata); state<=FC_WEIGHT; end
            FC_WEIGHT: if(input_fire) begin
                fc_lo<=$signed({1'b0,fc_input[index][15:0]})*$signed(s_axis_tdata[7:0]);
                fc_hi<=$signed(fc_input[index][31:16])*$signed(s_axis_tdata[7:0]);
                state<=FC_COMB;
            end
            FC_COMB: begin fc_product<=({{8{fc_hi[23]}},fc_hi} <<< 16)+{{7{fc_lo[24]}},fc_lo}; state<=FC_ACC; end
            FC_ACC: begin
                if(index==h[1]-1) begin
                    m_axis_tdata<=acc+fc_product; index<=0; state<=EMIT;
                end else begin acc<=acc+fc_product; index<=index+1; state<=FC_WEIGHT; end
            end
            EMIT: if(output_fire) begin
                if(final_word) begin state<=HEADER; packets<=packets+1; end
                else if(h[0]==0) begin
                    rr<=0; state<=REDUCE;
                    if(ox==out_width-1) begin
                        ox<=0;
                        if(oj==rows-1) begin oj<=0; og<=og+1; end
                        else oj<=oj+1;
                    end else ox<=ox+1;
                end else if(h[0]==1) begin index<=index+1; state<=ADD_A; end
                else begin
                    group_index<=group_index+1;
                    state<=h[0]==2 ? POOL : FC_BIAS;
                end
            end
            default: state<=FAULT;
            endcase
        end
    end
    // Ingress TLAST is transport framing only: commands may span DMA transfers.
    // Packet lengths come from validated headers; no dependence on TLAST timing.
endmodule
