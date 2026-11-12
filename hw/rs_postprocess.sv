// Six registered stages, four independent <=17x17 multiplications. Avoids a
// combinational 32x32 DSP cascade followed by an adder and barrel shifter.
// Valid shifts 1..62; ties round towards +infinity, including negative inputs.
module rs_postprocess (
    input wire clk, input wire resetn, input wire valid_in,
    input wire signed [31:0] value,
    input wire signed [31:0] multiplier,
    input wire [5:0] shift,
    input wire relu,
    output reg valid_out, output reg signed [31:0] result
);
    reg [4:0] valid;
    reg [5:0] sh [0:4];
    reg [4:0] re;
    (* use_dsp="yes" *) reg signed [33:0] p00;
    (* use_dsp="yes" *) reg signed [32:0] p01,p10;
    (* use_dsp="yes" *) reg signed [31:0] p11;
    reg signed [33:0] middle;
    reg signed [63:0] base,product,rounded,shifted;
    integer i;
    always @(posedge clk) begin
        if(!resetn) begin valid<=0; valid_out<=0; result<=0; re<=0; end
        else begin
            valid<={valid[3:0],valid_in}; valid_out<=valid[4];
            sh[0]<=shift; re<={re[3:0],relu};
            for(i=1;i<5;i=i+1) sh[i]<=sh[i-1];
            p00<=$signed({1'b0,value[15:0]})*$signed({1'b0,multiplier[15:0]});
            p01<=$signed({1'b0,value[15:0]})*$signed({1'b0,multiplier[30:16]});
            p10<=$signed(value[31:16])*$signed({1'b0,multiplier[15:0]});
            p11<=$signed(value[31:16])*$signed({1'b0,multiplier[30:16]});
            middle<={{1{p01[32]}},p01}+{{1{p10[32]}},p10};
            base<={{30{p00[33]}},p00}+{p11,32'd0};
            product<=base+{{14{middle[33]}},middle,16'd0};
            rounded<=product+(64'sd1 <<< (sh[2]-1));
            shifted<=rounded >>> sh[3];
            result<=shifted>127 ? 32'sd127 :
                    shifted<(re[4] ? 0 : -128) ? (re[4] ? 32'sd0 : -32'sd128) : shifted[31:0];
        end
    end
endmodule
