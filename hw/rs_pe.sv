// One stationary filter/activation row pair. No reset of distributed memories:
// controller loads rows/weights and clears each psum address on channel zero.
module rs_pe #(
    parameter integer MAX_WIDTH = 34,
    parameter integer MAX_OUT = 32,
    parameter integer MAX_KERNEL = 3,
    parameter integer P_CONTEXTS = 1,
    parameter integer Q_CONTEXTS = 1,
    parameter integer DOUBLE_BUFFER = 0,
    parameter integer PIPELINE = 0,
    parameter integer EXTERNAL_ACT = 0
) (
    input wire clk, input wire resetn,
    input wire signed [7:0] shared_activation,output wire signed [7:0] activation_value,
    input wire act_we, input wire [7:0] act_addr,
    input wire signed [7:0] act_data,
    input wire weight_we, input wire [1:0] weight_addr,
    input wire signed [7:0] weight_data,
    input wire sample_en, input wire mac_en, input wire first_mac,
    input wire [7:0] sample_addr, input wire [1:0] tap,
    input wire [7:0] out_addr,
    input wire [7:0] read_addr,
    output wire signed [31:0] psum,
    input wire reduce_en, input wire signed [31:0] neighbor_psum,
    output reg signed [31:0] link_psum,
    input wire load_bank, compute_bank,
    input wire [7:0] load_p, load_q, compute_p, sample_p, sample_q, read_p
);
    localparam integer PB=P_CONTEXTS<2 ? 1 : $clog2(P_CONTEXTS);
    localparam integer QB=Q_CONTEXTS<2 ? 1 : $clog2(Q_CONTEXTS);
    localparam integer BANKS=DOUBLE_BUFFER ? 2 : 1;
    (* ram_style = "distributed" *) reg signed [7:0] acts [0:BANKS*Q_CONTEXTS*MAX_WIDTH-1];
    reg signed [7:0] weights [0:BANKS*P_CONTEXTS*Q_CONTEXTS*MAX_KERNEL-1];
    (* ram_style = "distributed" *) reg signed [31:0] sums [0:P_CONTEXTS*MAX_OUT-1];
    wire [15:0] load_act=((DOUBLE_BUFFER ? load_bank : 0)*Q_CONTEXTS+(Q_CONTEXTS==1 ? 0 : load_q[QB-1:0]))*MAX_WIDTH+act_addr;
    wire [15:0] sample_act=((DOUBLE_BUFFER ? compute_bank : 0)*Q_CONTEXTS+(Q_CONTEXTS==1 ? 0 : sample_q[QB-1:0]))*MAX_WIDTH+sample_addr;
    wire [15:0] load_weight=(((DOUBLE_BUFFER ? load_bank : 0)*P_CONTEXTS+(P_CONTEXTS==1 ? 0 : load_p[PB-1:0]))*Q_CONTEXTS+(Q_CONTEXTS==1 ? 0 : load_q[QB-1:0]))*MAX_KERNEL+weight_addr;
    wire [15:0] sample_weight=(((DOUBLE_BUFFER ? compute_bank : 0)*P_CONTEXTS+(P_CONTEXTS==1 ? 0 : sample_p[PB-1:0]))*Q_CONTEXTS+(Q_CONTEXTS==1 ? 0 : sample_q[QB-1:0]))*MAX_KERNEL+tap;
    wire [15:0] write_sum=(P_CONTEXTS==1 ? 0 : compute_p[PB-1:0])*MAX_OUT+out_addr;
    wire [15:0] read_sum=(P_CONTEXTS==1 ? 0 : read_p[PB-1:0])*MAX_OUT+read_addr;
    (* use_dsp="yes" *) reg signed [15:0] product;
    wire signed [31:0] extended_product = {{16{product[15]}},product};
    reg [15:0] registered_act,registered_weight;
    reg address_valid;
    generate if(PIPELINE==2) begin: address_pipe
        always @(posedge clk) begin
            if(!resetn) address_valid<=0;
            else begin
                address_valid<=sample_en;
                if(sample_en) begin registered_act<=sample_act;registered_weight<=sample_weight;end
            end
        end
    end endgenerate
    wire [15:0] act_lookup=PIPELINE==2 ? registered_act : sample_act;
    wire [15:0] weight_lookup=PIPELINE==2 ? registered_weight : sample_weight;
    wire operand_valid=PIPELINE==2 ? address_valid : sample_en;
    assign activation_value=EXTERNAL_ACT ? shared_activation : acts[act_lookup];
    wire effective_mac,effective_first;
    wire [15:0] effective_sum;
    generate if(PIPELINE) begin: pipe
        reg signed [7:0] activation,weight;
        reg sampled,commit_en,commit_first,commit_en2,commit_first2;
        reg [15:0] commit_sum,commit_sum2;
        always @(posedge clk) begin
            if(!resetn) begin sampled<=0;commit_en<=0;commit_en2<=0;end
            else begin
                sampled<=operand_valid;commit_en<=mac_en;
                commit_en2<=commit_en;commit_first2<=commit_first;commit_sum2<=commit_sum;
                commit_first<=first_mac;commit_sum<=write_sum;
                if(operand_valid) begin activation<=activation_value;weight<=weights[weight_lookup];end
                if(sampled) product<=activation*weight;
            end
        end
        assign effective_mac=PIPELINE==2 ? commit_en2 : commit_en;
        assign effective_first=PIPELINE==2 ? commit_first2 : commit_first;
        assign effective_sum=PIPELINE==2 ? commit_sum2 : commit_sum;
    end else begin: direct
        always @(posedge clk) if(sample_en) product<=activation_value*weights[sample_weight];
        assign effective_mac=mac_en;
        assign effective_first=first_mac;
        assign effective_sum=write_sum;
    end endgenerate
    always @(posedge clk) begin
        if (act_we) acts[load_act] <= act_data;
        if (weight_we) weights[load_weight] <= weight_data;
        if (effective_mac) sums[effective_sum] <= (effective_first ? 32'sd0 : sums[effective_sum]) + extended_product;
        if (reduce_en) link_psum <= sums[read_sum] + neighbor_psum;
    end
    assign psum = sums[read_sum];
endmodule
