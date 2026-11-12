// Two-bank synchronous 1R/1W global ingress buffer. One word per cycle, no
// combinational RAM reads, one reserved output register, explicit backpressure.
module rs_axis_buffer #(
    parameter integer BANK_ADDR_BITS=8
) (
    input wire clk, input wire resetn,
    input wire [31:0] s_data, input wire s_valid, output wire s_ready,
    output wire [31:0] m_data, output wire m_valid, input wire m_ready
);
    localparam integer DEPTH=2*(1<<BANK_ADDR_BITS);
    reg [BANK_ADDR_BITS:0] wp, rp;
    reg [BANK_ADDR_BITS+1:0] count;
    reg valid, read_bank;
    wire push=s_valid && s_ready;
    wire fetch=(count!=0) && (!valid || m_ready);
    wire [31:0] q [0:1];
    assign s_ready=resetn && count<DEPTH;
    assign m_valid=resetn && valid;
    assign m_data=q[read_bank];
    genvar bank;
    generate for(bank=0;bank<2;bank=bank+1) begin: banks
        (* ram_style="block" *) reg [31:0] mem [0:(1<<BANK_ADDR_BITS)-1];
        reg [31:0] rd;
        always @(posedge clk) begin
            if(push && wp[0]==bank) mem[wp[BANK_ADDR_BITS:1]]<=s_data;
            if(fetch && rp[0]==bank) rd<=mem[rp[BANK_ADDR_BITS:1]];
        end
        assign q[bank]=rd;
    end endgenerate
    always @(posedge clk) begin
        if(!resetn) begin wp<=0; rp<=0; count<=0; valid<=0; read_bank<=0; end
        else begin
            case({push,fetch})
                2'b10: count<=count+1;
                2'b01: count<=count-1;
                default: count<=count;
            endcase
            if(push) wp<=wp+1;
            if(fetch) begin rp<=rp+1; read_bank<=rp[0]; valid<=1; end
            else if(m_ready) valid<=0;
        end
    end
endmodule
