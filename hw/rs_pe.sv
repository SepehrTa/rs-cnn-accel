// One stationary filter/activation row pair. No reset of distributed memories:
// controller loads rows/weights and clears each psum address on channel zero.
module rs_pe #(
    parameter integer MAX_WIDTH = 34,
    parameter integer MAX_OUT = 32,
    parameter integer MAX_KERNEL = 3
) (
    input wire clk,
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
    output reg signed [31:0] link_psum
);
    (* ram_style = "distributed" *) reg signed [7:0] acts [0:MAX_WIDTH-1];
    reg signed [7:0] weights [0:MAX_KERNEL-1];
    (* ram_style = "distributed" *) reg signed [31:0] sums [0:MAX_OUT-1];
    (* use_dsp="yes" *) reg signed [15:0] product;
    wire signed [31:0] extended_product = {{16{product[15]}},product};
    always @(posedge clk) begin
        if (sample_en) product <= acts[sample_addr] * weights[tap];
        if (act_we) acts[act_addr] <= act_data;
        if (weight_we) weights[weight_addr] <= weight_data;
        if (mac_en) sums[out_addr] <= (first_mac ? 32'sd0 : sums[out_addr]) + extended_product;
        if (reduce_en) link_psum <= sums[read_addr] + neighbor_psum;
    end
    assign psum = sums[read_addr];
endmodule
