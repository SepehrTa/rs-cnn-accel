// ABI-2 tile codec. Four little-endian signed INT8 payload/results per word;
// 16-word headers, classifier inputs/biases and pool/FC outputs remain INT32.
// Header[15].bit0 selects packing; zero retains the original scalar protocol.
// Byte segment tails use zero padding and all four AXIS byte lanes are valid.
module rs_packet_codec (
    input wire clk, resetn,
    input wire [31:0] s_data, input wire s_valid, output wire s_ready,
    output wire [31:0] core_s_data, output wire core_s_valid, input wire core_s_ready,
    input wire [31:0] core_m_data, input wire core_m_valid, core_m_last,
    output wire core_m_ready,
    output reg [31:0] m_data, output reg m_valid, m_last, input wire m_ready,
    output wire busy,
    output reg [31:0] input_words, output_words
);
    localparam HEADER=0, SETUP1=1, SETUP2=2, BYTES=3, WORDS=4, WAIT_OUTPUT=5, SETUP0=6, SETUP3=7;
    reg [2:0] phase;
    reg [3:0] hi;
    reg [31:0] h[0:15];
    reg packed_mode;
    reg [31:0] remaining, class_index;
    reg [1:0] fc_phase;
    reg [15:0] channel_size,weight_count;
    reg [7:0] registered_rows;
    (* use_dsp="yes" *) reg [15:0] act_count;
    reg [31:0] byte_word;
    reg byte_valid;
    reg [1:0] byte_index;
    wire [7:0] kernel=h[1][7:0];
    wire [7:0] patch_rows=(h[2]==2 ? ((h[5][7:0]-1)<<1) : h[5][7:0]-1)+kernel;
    wire byte_mode=phase==BYTES && packed_mode;
    wire scalar_valid=(phase==HEADER || phase==WORDS || phase==BYTES) && !byte_mode;
    wire scalar_fire=core_s_valid && core_s_ready;
    wire byte_end=byte_index==3 || remaining==1;
    wire [7:0] byte_value=byte_word >> (8*byte_index);
    assign core_s_data=byte_mode ? {{24{byte_value[7]}},byte_value} : s_data;
    assign core_s_valid=resetn && (byte_mode ? byte_valid : scalar_valid && s_valid);
    assign s_ready=resetn && (byte_mode ?
        (!byte_valid || (core_s_ready && byte_end && remaining!=1)) : scalar_valid && core_s_ready);
    assign busy=phase!=HEADER || hi!=0 || m_valid;
    wire pack_result=packed_mode && (h[0]==0 || h[0]==1);
    reg [31:0] result_word;
    reg [1:0] result_index;
    wire [31:0] assembled=result_word | ({24'd0,core_m_data[7:0]} << (8*result_index));
    assign core_m_ready=resetn && (!m_valid || m_ready);
    integer i;
    always @(posedge clk) begin
        if(!resetn) begin
            phase<=HEADER; hi<=0; packed_mode<=0; remaining<=0;
            class_index<=0; fc_phase<=0; channel_size<=0;
            byte_word<=0; byte_valid<=0; byte_index<=0;
            weight_count<=0;registered_rows<=0;act_count<=0;
            m_data<=0; m_valid<=0; m_last<=0; result_word<=0; result_index<=0;
            input_words<=0; output_words<=0;
            for(i=0;i<16;i=i+1) h[i]<=0;
        end else begin
            if(s_valid && s_ready) input_words<=input_words+1;
            if(m_valid && m_ready) begin
                m_valid<=0; output_words<=output_words+1;
                if(m_last) begin phase<=HEADER; hi<=0; end
            end
            if(core_m_valid && core_m_ready) begin
                if(pack_result) begin
                    if(result_index==3 || core_m_last) begin
                        m_data<=assembled; m_last<=core_m_last; m_valid<=1;
                        result_index<=0; result_word<=0;
                    end else begin result_word<=assembled; result_index<=result_index+1; end
                end else begin m_data<=core_m_data; m_last<=core_m_last; m_valid<=1; end
            end
            if(byte_mode) begin
                if(s_valid && s_ready) begin
                    byte_word<=s_data; byte_valid<=1; byte_index<=0;
                end else if(scalar_fire && byte_end) begin byte_valid<=0; byte_index<=0; end
                else if(scalar_fire) byte_index<=byte_index+1;
            end
            case(phase)
            HEADER: if(scalar_fire) begin
                h[hi]<=s_data;
                if(hi==15) begin packed_mode<=s_data[0]; hi<=0; phase<=SETUP0; end
                else hi<=hi+1;
            end
            SETUP0: begin
                registered_rows<=patch_rows;
                weight_count<=kernel==3 ? ({8'd0,h[6][7:0]}<<3)+h[6][7:0] : h[6][7:0];
                phase<=SETUP1;
            end
            SETUP1: begin act_count<=registered_rows*h[3][7:0];phase<=SETUP3;end
            SETUP3: begin
                channel_size<=act_count+weight_count;
                class_index<=0; fc_phase<=0; byte_valid<=0; byte_index<=0;
                phase<=SETUP2;
            end
            SETUP2: begin
                if(h[0]==0) begin
                    if(h[6]>2) begin remaining<=(h[6]-2)*3; phase<=WORDS; end
                    else begin remaining<=h[7][15:0]*channel_size; phase<=BYTES; end
                end
                else if(h[0]==1) begin remaining<=h[1]<<1; phase<=BYTES; end
                else if(h[0]==2) begin remaining<=h[1][15:0]*h[2][15:0]; phase<=BYTES; end
                else if(h[0]==3) begin remaining<=h[1]; phase<=WORDS; end
                else phase<=WAIT_OUTPUT;
            end
            BYTES, WORDS: if(scalar_fire) begin
                remaining<=remaining-1;
                if(remaining==1) begin
                    byte_valid<=0; byte_index<=0;
                    if(h[0]==0 && phase==WORDS) begin remaining<=h[7][15:0]*channel_size; phase<=BYTES; end
                    else if(h[0]!=3) phase<=WAIT_OUTPUT;
                    else if(fc_phase==0) begin remaining<=1; fc_phase<=1; phase<=WORDS; end
                    else if(fc_phase==1) begin remaining<=h[1]; fc_phase<=2; phase<=BYTES; end
                    else if(class_index==h[2]-1) phase<=WAIT_OUTPUT;
                    else begin class_index<=class_index+1; remaining<=1; fc_phase<=1; phase<=WORDS; end
                end
            end
            default: begin end
            endcase
        end
    end
endmodule
