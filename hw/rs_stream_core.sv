// Packet-driven RS engine. All state advances only on accepted words or compute
// clocks. AXIS output is stable under backpressure. See docs/ARCHITECTURE.md.
module rs_stream_core #(
    parameter integer FILTERS = 2,
    parameter integer COLS = 8,
    parameter integer KMAX = 3,
    parameter integer MAX_WIDTH = 34,
    parameter integer MAX_OUT = 32,
    parameter integer MAX_FC = 256,
    parameter integer P_CONTEXTS = 1,
    parameter integer Q_CONTEXTS = 1
) (
    input wire aclk, input wire aresetn,
    input wire [31:0] s_axis_tdata, input wire s_axis_tvalid,
    output wire s_axis_tready, input wire s_axis_tlast,
    output reg [31:0] m_axis_tdata, output wire m_axis_tvalid,
    input wire m_axis_tready, output wire m_axis_tlast,
    output wire busy, output reg error,
    output reg [31:0] cycles, output reg [31:0] packets,
    output reg [31:0] macs, load_wait_cycles, overlap_cycles
);
    localparam HEADER=0, CHECK=1, ACT=2, WEIGHT=3, MAC=4,
               REDUCE=5, QUANT=6, EMIT=7, ADD_A=8, ADD_B=9,
               ADD_CALC=10, POOL=11, FC_INPUT=12, FC_BIAS=13,
               FC_WEIGHT=14, FAULT=15, POST_START=16, POST_WAIT=17,
               MAC_FETCH=18, ADD_SUM=19, ADD_SHIFT=20, FC_COMB=21, FC_ACC=22,
               BANK_WAIT=23, PARAM=24, MAC_DRAIN=25, MAC_DRAIN2=26, CONV_SUM=27;
    localparam LD_IDLE=0, LD_ACT=1, LD_WEIGHT=2, LD_WAIT=3;
    localparam MAX_FILTERS=FILTERS*P_CONTEXTS;
    reg [4:0] state;
    reg [31:0] h [0:15];
    reg [3:0] hi;
    reg [7:0] iy, ix, wg, wr, wt, ox, tap, og, oj, rr;
    reg [31:0] channel, index, group_index;
    reg [1:0] ld_state;
    reg [31:0] ld_channel;
    reg ld_bank,compute_bank;
    reg [1:0] bank_ready;
    localparam PB=P_CONTEXTS<2 ? 1 : $clog2(P_CONTEXTS);
    localparam QB=Q_CONTEXTS<2 ? 1 : $clog2(Q_CONTEXTS);
    reg [QB-1:0] ld_q,cq;
    reg [PB-1:0] cp;
    reg [7:0] param_filter;
    reg [$clog2(P_CONTEXTS+1)-1:0] p_count;
    reg [$clog2(Q_CONTEXTS+1)-1:0] q_count;
    reg [15:0] mac_steps[0:P_CONTEXTS-1];
    reg [1:0] param_kind;
    reg signed [31:0] biases[0:MAX_FILTERS-1],multipliers[0:MAX_FILTERS-1];
    reg [5:0] shifts[0:MAX_FILTERS-1];
    reg signed [31:0] acc, operand_a;
    localparam RB=FILTERS*KMAX*COLS<2 ? 1 : $clog2(FILTERS*KMAX*COLS);
    localparam CB=COLS<2 ? 1 : $clog2(COLS);
    reg [RB-1:0] reduction_index;
    (* dont_touch="true" *) reg signed [31:0] selected_link;
    reg signed [31:0] fc_input [0:MAX_FC-1];
    wire [7:0] k = h[1][7:0], stride = h[2][7:0];
    wire [7:0] width = h[3][7:0], out_width = h[4][7:0];
    wire [7:0] rows = h[5][7:0], filters = h[6][7:0];
    wire [7:0] patch_rows = (stride==2 ? ((rows-1)<<1) : rows-1)+k;
    // Runtime stride has two legal values. A generic 32x32 validation multiply
    // here used to dominate timing despite never participating in inference.
    wire [8:0] required_width = (stride==2 ? {out_width,1'b0}-9'd2 : {1'b0,out_width}-9'd1)+{1'b0,k};
    // The multiplier register consumes the next sample while the previous
    // product updates its psum. One priming clock per channel, then one MAC
    // per active PE per clock, including adjacent taps to the same psum.
    // Runtime dimensions are latched outside the per-MAC critical path.
    wire [15:0] active_rows=k==3 ? ({8'd0,rows}<<1)+rows : {8'd0,rows};
    function [15:0] mac_step(input integer context_index);
        reg [7:0] groups;
        begin
            groups=filters>=((context_index+1)*FILTERS) ? FILTERS :
                   filters>context_index*FILTERS ? filters-context_index*FILTERS : 0;
            case(groups)
            1:mac_step=active_rows;
            2:mac_step=active_rows<<1;
            3:mac_step=(active_rows<<1)+active_rows;
            4:mac_step=active_rows<<2;
            default:mac_step=0;
            endcase
        end
    endfunction
    wire last_p=cp==p_count-1'b1;
    wire last_q=cq==q_count-1'b1;
    wire final_mac = last_p && last_q && tap==k-1 && ox==out_width-1;
    wire [PB-1:0] next_p=last_p ? 8'd0 : cp+8'd1;
    wire [QB-1:0] next_q=last_p ? (last_q ? 8'd0 : cq+8'd1) : cq;
    wire [7:0] next_tap = last_p && last_q ? (tap==k-1 ? 8'd0 : tap+8'd1) : tap;
    wire [7:0] next_ox = last_p && last_q && tap==k-1 ? ox+8'd1 : ox;
    wire [PB-1:0] fetch_p=state==MAC_FETCH ? cp : next_p;
    wire [QB-1:0] fetch_q=state==MAC_FETCH ? cq : next_q;
    wire [7:0] fetch_ox = state==MAC_FETCH ? ox : next_ox;
    wire [7:0] fetch_tap = state==MAC_FETCH ? tap : next_tap;
    wire input_fire = s_axis_tvalid && s_axis_tready;
    wire output_fire = m_axis_tvalid && m_axis_tready;
    wire last_conv = og==filters-1 && oj==rows-1 && ox==out_width-1;
    wire final_word = h[0]==0 ? last_conv :
                      h[0]==1 ? index==h[1]-1 : group_index==h[2]-1;
    assign s_axis_tready = aresetn && (state==HEADER || state==PARAM || ld_state==LD_ACT || ld_state==LD_WEIGHT ||
                           state==ADD_A || state==ADD_B || state==POOL ||
                           state==FC_INPUT || state==FC_BIAS || state==FC_WEIGHT);
    assign m_axis_tvalid = aresetn && state==EMIT;
    assign m_axis_tlast = m_axis_tvalid && final_word;
    assign busy = state!=HEADER || hi!=0;

    wire signed [31:0] pe_sums [0:FILTERS*KMAX*COLS-1];
    wire signed [31:0] pe_links [0:FILTERS*KMAX*COLS-1];
    wire signed [7:0] row_activations [0:KMAX*COLS-1];
    genvar g,r,j;
    generate for(g=0;g<FILTERS;g=g+1) begin: groups
        for(r=0;r<KMAX;r=r+1) begin: prows
            for(j=0;j<COLS;j=j+1) begin: pcols
                localparam integer P=(g*KMAX+r)*COLS+j;
                wire signed [31:0] neighbor_psum;
                wire signed [7:0] activation_value;
                if(g==0) assign row_activations[r*COLS+j]=activation_value;
                if(r==0) assign neighbor_psum=32'sd0;
                else assign neighbor_psum=pe_links[(g*KMAX+r-1)*COLS+j];
                // Diagonal multicast: identical source row is written to all
                // row pairs requiring iy=j*stride+r, and both filter groups.
                rs_pe #(.MAX_WIDTH(MAX_WIDTH),.MAX_OUT(MAX_OUT),.MAX_KERNEL(KMAX),
                    .P_CONTEXTS(P_CONTEXTS),.Q_CONTEXTS(Q_CONTEXTS),.DOUBLE_BUFFER(1),.PIPELINE(2),.EXTERNAL_ACT(g!=0)) pe (
                    .clk(aclk),.resetn(aresetn),
                    .shared_activation(row_activations[r*COLS+j]),.activation_value(activation_value),
                    .act_we(input_fire && ld_state==LD_ACT && j<rows && r<k && g==0 && iy==j*stride+r),
                    .act_addr(ix), .act_data(s_axis_tdata[7:0]),
                    .weight_we(input_fire && ld_state==LD_WEIGHT && g==wg%FILTERS && r==wr && j<rows),
                    .weight_addr(wt[1:0]), .weight_data(s_axis_tdata[7:0]),
                    .sample_en((state==MAC_FETCH || (state==MAC && !final_mac)) && g+fetch_p*FILTERS<filters && r<k && j<rows),
                    .mac_en(state==MAC && g+cp*FILTERS<filters && r<k && j<rows),
                    .first_mac(channel==0 && cq==0 && tap==0),
                    .sample_addr((stride==2 ? (fetch_ox<<1) : fetch_ox)+fetch_tap), .tap(fetch_tap[1:0]),
                    .out_addr(ox), .read_addr(ox), .psum(pe_sums[P]),
                    .reduce_en(state==REDUCE && g==og%FILTERS && j==oj && r==rr),
                    .neighbor_psum(neighbor_psum),.link_psum(pe_links[P]),
                    .load_bank(ld_bank),.compute_bank(compute_bank),.load_p(wg/FILTERS),.load_q(ld_q),
                    .compute_p(cp),.sample_p(fetch_p),.sample_q(fetch_q),.read_p(og/FILTERS)
                );
            end
        end
    end endgenerate

    wire signed [31:0] multiplier = multipliers[og];
    wire [5:0] shift = shifts[og];
    wire signed [31:0] bias = biases[og];
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
            reduction_index<=0;selected_link<=0;
            macs<=0; load_wait_cycles<=0; overlap_cycles<=0;p_count<=0;q_count<=0;
            for(z=0;z<P_CONTEXTS;z=z+1) mac_steps[z]<=0;
            ld_state<=LD_IDLE; ld_channel<=0; ld_q<=0; ld_bank<=0; compute_bank<=0;
            bank_ready<=0; cp<=0; cq<=0; param_filter<=0; param_kind<=0;
            m_axis_tdata<=0; channel<=0; index<=0; group_index<=0;
            iy<=0; ix<=0; wg<=0; wr<=0; wt<=0; ox<=0; tap<=0;
            og<=0; oj<=0; rr<=0; acc<=0; operand_a<=0; add_sum<=0;
            for(z=0;z<16;z=z+1) h[z]<=0;
            for(z=0;z<MAX_FILTERS;z=z+1) begin biases[z]<=0; multipliers[z]<=0; shifts[z]<=0; end
        end else begin
            if (busy) cycles<=cycles+1;
            if(state==MAC) macs<=macs+mac_steps[cp];
            if(state==BANK_WAIT && !bank_ready[compute_bank]) load_wait_cycles<=load_wait_cycles+1;
            if(state==MAC && input_fire && (ld_state==LD_ACT || ld_state==LD_WEIGHT)) overlap_cycles<=overlap_cycles+1;
            // Independently fill the inactive bank; publish it atomically once
            // all Q channel rows and P filter contexts have arrived.
            case(ld_state)
            LD_WAIT: if(!bank_ready[ld_bank]) ld_state<=LD_ACT;
            LD_ACT: if(input_fire) begin
                if(ix==width-1) begin
                    ix<=0;
                    if(iy==patch_rows-1) begin iy<=0; ld_state<=LD_WEIGHT; end
                    else iy<=iy+1;
                end else ix<=ix+1;
            end
            LD_WEIGHT: if(input_fire) begin
                if(wt==k-1) begin
                    wt<=0;
                    if(wr==k-1) begin
                        wr<=0;
                        if(wg==filters-1) begin
                            wg<=0;
                            if(ld_q==Q_CONTEXTS-1 || ld_channel==h[7]-1) begin
                                bank_ready[ld_bank]<=1; ld_q<=0;
                                if(ld_channel==h[7]-1) ld_state<=LD_IDLE;
                                else begin ld_bank<=!ld_bank; ld_state<=LD_WAIT; end
                            end else begin ld_q<=ld_q+1; ld_state<=LD_ACT; end
                            ld_channel<=ld_channel+1;
                        end else wg<=wg+1;
                    end else wr<=wr+1;
                end else wt<=wt+1;
            end
            default: begin end
            endcase
            case(state)
            HEADER: if(input_fire) begin
                h[hi]<=s_axis_tdata;
                if(hi==15) begin hi<=0; state<=CHECK; end
                else hi<=hi+1;
            end
            CHECK: begin
                p_count<=(filters+FILTERS-1)/FILTERS;
                for(z=0;z<P_CONTEXTS;z=z+1) mac_steps[z]<=mac_step(z);
                channel<=0; index<=0; group_index<=0; iy<=0; ix<=0;
                wg<=0; wr<=0; wt<=0; ox<=0; tap<=0; og<=0; oj<=0; acc<=0;
                ld_state<=LD_IDLE; ld_channel<=0; ld_q<=0; ld_bank<=0; compute_bank<=0;
                bank_ready<=0; cp<=0; cq<=0; param_filter<=2; param_kind<=0;
                biases[0]<=h[9]; multipliers[0]<=h[10]; shifts[0]<=h[11][5:0];
                if(MAX_FILTERS>1) begin biases[1]<=h[12]; multipliers[1]<=h[13]; shifts[1]<=h[14][5:0]; end
                if(h[0]==0 && (h[1]==1 || h[1]==3) && h[1]<=KMAX &&
                   (h[2]==1 || h[2]==2) && h[3]>=h[1] && h[3]<=MAX_WIDTH &&
                   h[4]>0 && h[4]<=MAX_OUT && required_width<={1'b0,width} &&
                   h[5]>0 && h[5]<=COLS && h[6]>0 && h[6]<=MAX_FILTERS &&
                   h[7]>0 && h[7]<=65535 && h[10]>0 && h[10]<32'h80000000 &&
                   h[11]>=1 && h[11]<=62 && (h[6]==1 ||
                   (h[13]>0 && h[13]<32'h80000000 && h[14]>=1 && h[14]<=62))) begin
                    if(h[6]>2) state<=PARAM;
                    else begin state<=BANK_WAIT; ld_state<=LD_ACT; end
                end
                else if(h[0]==1 && h[1]>0 && h[10]<32'h80000000 &&
                        h[13]<32'h80000000 && h[11]>=1 && h[11]<=62) state<=ADD_A;
                else if(h[0]==2 && h[1]>0 && h[1]<=65535 && h[2]>0 && h[2]<=65535) state<=POOL;
                else if(h[0]==3 && h[1]>0 && h[1]<=MAX_FC && h[2]>0 && h[2]<=65535) state<=FC_INPUT;
                else begin state<=FAULT; error<=1; end
            end
            PARAM: if(input_fire) begin
                case(param_kind)
                0: begin biases[param_filter]<=s_axis_tdata; param_kind<=1; end
                1: begin
                    if(s_axis_tdata==0 || s_axis_tdata[31]) begin state<=FAULT; error<=1; end
                    else begin multipliers[param_filter]<=s_axis_tdata; param_kind<=2; end
                end
                2: begin
                    if(s_axis_tdata<1 || s_axis_tdata>62) begin state<=FAULT; error<=1; end
                    else begin
                        shifts[param_filter]<=s_axis_tdata[5:0]; param_kind<=0;
                        if(param_filter==filters-1) begin state<=BANK_WAIT; ld_state<=LD_ACT; end
                        else param_filter<=param_filter+1;
                    end
                end
                endcase
            end
            BANK_WAIT: begin
                q_count<=h[7]-channel<Q_CONTEXTS ? h[7]-channel : Q_CONTEXTS;
                if(bank_ready[compute_bank]) begin ox<=0; tap<=0; cp<=0; cq<=0; state<=MAC_FETCH; end
            end
            MAC_FETCH: state<=MAC;
            MAC: begin
                cp<=next_p; cq<=next_q; tap<=next_tap; ox<=next_ox;
                if(final_mac) begin
                    bank_ready[compute_bank]<=0;
                    ox<=0; tap<=0; cp<=0; cq<=0;
                    if(channel+q_count==h[7]) begin rr<=0; state<=MAC_DRAIN; end
                    else begin channel<=channel+q_count; compute_bank<=!compute_bank; state<=BANK_WAIT; end
                end
            end
            MAC_DRAIN: state<=MAC_DRAIN2;
            MAC_DRAIN2: state<=REDUCE;
            REDUCE: begin
                reduction_index<=((og%FILTERS)*KMAX+(k==3 ? 2 : 0))*COLS+oj[CB-1:0];
                // Enable one directly wired neighbor hop per clock. Arithmetic
                // and link storage are inside each PE, never a shared reducer.
                if(rr==k-1) state<=QUANT;
                else rr<=rr+1;
            end
            QUANT: begin selected_link<=pe_links[reduction_index];state<=CONV_SUM;end
            CONV_SUM: begin acc<=selected_link+bias;state<=POST_START;end
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
