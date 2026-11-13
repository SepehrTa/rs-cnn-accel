// AXI4-Lite control + AXIS data plane. Single clock domain, PS FCLK0 at 100MHz.
module rs_accel_top #(
    parameter integer COLS=8,
    parameter integer BANK_ADDR_BITS=8,
    parameter integer FILTERS=2,
    parameter integer P_CONTEXTS=1,
    parameter integer Q_CONTEXTS=1
) (
    (* X_INTERFACE_PARAMETER="ASSOCIATED_BUSIF s_axi:s_axis:m_axis, ASSOCIATED_RESET aresetn, FREQ_HZ 100000000" *)
    (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 aclk CLK" *) input wire aclk,
    (* X_INTERFACE_PARAMETER="POLARITY ACTIVE_LOW" *)
    (* X_INTERFACE_INFO="xilinx.com:signal:reset:1.0 aresetn RST" *) input wire aresetn,
    input wire [7:0] s_axi_awaddr, input wire s_axi_awvalid, output wire s_axi_awready,
    input wire [31:0] s_axi_wdata, input wire [3:0] s_axi_wstrb,
    input wire s_axi_wvalid, output wire s_axi_wready,
    output reg [1:0] s_axi_bresp, output reg s_axi_bvalid, input wire s_axi_bready,
    input wire [7:0] s_axi_araddr, input wire s_axi_arvalid, output wire s_axi_arready,
    output reg [31:0] s_axi_rdata, output reg [1:0] s_axi_rresp,
    output reg s_axi_rvalid, input wire s_axi_rready,
    input wire [31:0] s_axis_tdata, input wire [3:0] s_axis_tkeep,
    input wire s_axis_tvalid, output wire s_axis_tready, input wire s_axis_tlast,
    output wire [31:0] m_axis_tdata, output wire [3:0] m_axis_tkeep,
    output wire m_axis_tvalid, input wire m_axis_tready, output wire m_axis_tlast
);
    reg have_aw, have_w;
    reg [7:0] awaddr;
    reg [31:0] wdata;
    reg [3:0] wstrb;
    reg soft_reset;
    wire core_resetn=aresetn && !soft_reset;
    wire busy,error,core_error,resident_busy,resident_error,model_valid;
    wire [31:0] cycles,packets,buffer_data;
    wire buffer_valid,buffer_ready;
    wire [31:0] core_input,core_output,stream_inputs,stream_outputs,decoded_input;
    wire decoded_valid,decoded_ready;
    wire core_input_valid,core_input_ready,core_output_valid,core_output_ready,core_output_last,codec_busy;
    wire [31:0] mac_count,load_wait,load_overlap;
    wire [31:0] router_data,codec_data,codec_inputs,codec_outputs;
    wire router_valid,router_ready,codec_valid,codec_ready,codec_last;
    wire [31:0] network_cycles,model_reads,act_reads,act_write_bytes,images;
    wire [7:0] current_layer;
    reg [31:0] dma_inputs,dma_outputs;
    assign stream_inputs=dma_inputs;
    assign stream_outputs=dma_outputs;
    assign error=core_error || resident_error;
    rs_axis_buffer #(.BANK_ADDR_BITS(BANK_ADDR_BITS)) ingress(.clk(aclk),.resetn(core_resetn),
        .s_data(s_axis_tdata),.s_valid(s_axis_tvalid),.s_ready(s_axis_tready),
        .m_data(buffer_data),.m_valid(buffer_valid),.m_ready(buffer_ready));
    rs_resident #(.COLS(COLS),.FILTERS(FILTERS),.P_CONTEXTS(P_CONTEXTS)) resident(.clk(aclk),.resetn(core_resetn),
        .s_data(buffer_data),.s_valid(buffer_valid),.s_ready(buffer_ready),
        .c_data(router_data),.c_valid(router_valid),.c_ready(router_ready),
        .e_data(codec_data),.e_valid(codec_valid),.e_ready(codec_ready),.e_last(codec_last),
        .m_data(m_axis_tdata),.m_valid(m_axis_tvalid),.m_ready(m_axis_tready),.m_last(m_axis_tlast),
        .core_error(core_error),.busy(resident_busy),.error(resident_error),.model_valid(model_valid),
        .network_cycles(network_cycles),.model_reads(model_reads),.activation_reads(act_reads),
        .activation_write_bytes(act_write_bytes),.completed_images(images),.current_layer(current_layer));
    rs_packet_codec codec(.clk(aclk),.resetn(core_resetn),
        .s_data(router_data),.s_valid(router_valid),.s_ready(router_ready),
        .core_s_data(decoded_input),.core_s_valid(decoded_valid),.core_s_ready(decoded_ready),
        .core_m_data(core_output),.core_m_valid(core_output_valid),.core_m_ready(core_output_ready),.core_m_last(core_output_last),
        .m_data(codec_data),.m_valid(codec_valid),.m_ready(codec_ready),.m_last(codec_last),
        .busy(codec_busy),.input_words(codec_inputs),.output_words(codec_outputs));
    // Registered scalar FIFO breaks descriptor/cache-valid paths before the
    // high-fanout PE loader/control network, with one word/clock throughput.
    rs_axis_buffer #(.BANK_ADDR_BITS(2)) decoded_fifo(.clk(aclk),.resetn(core_resetn),
        .s_data(decoded_input),.s_valid(decoded_valid),.s_ready(decoded_ready),
        .m_data(core_input),.m_valid(core_input_valid),.m_ready(core_input_ready));
    rs_stream_core #(.COLS(COLS),.FILTERS(FILTERS),.P_CONTEXTS(P_CONTEXTS),.Q_CONTEXTS(Q_CONTEXTS)) core(.aclk(aclk),.aresetn(core_resetn),
        .s_axis_tdata(core_input),.s_axis_tvalid(core_input_valid),.s_axis_tready(core_input_ready),.s_axis_tlast(1'b0),
        .m_axis_tdata(core_output),.m_axis_tvalid(core_output_valid),.m_axis_tready(core_output_ready),.m_axis_tlast(core_output_last),
        .busy(busy),.error(core_error),.cycles(cycles),.packets(packets),
        .macs(mac_count),.load_wait_cycles(load_wait),.overlap_cycles(load_overlap));
    assign m_axis_tkeep=4'hf;
    assign s_axi_awready=aresetn && !have_aw && !s_axi_bvalid;
    assign s_axi_wready=aresetn && !have_w && !s_axi_bvalid;
    assign s_axi_arready=aresetn && !s_axi_rvalid;
    always @(posedge aclk) begin
        if(!core_resetn) begin dma_inputs<=0;dma_outputs<=0;end
        else begin
            if(s_axis_tvalid && s_axis_tready) dma_inputs<=dma_inputs+1;
            if(m_axis_tvalid && m_axis_tready) dma_outputs<=dma_outputs+1;
        end
        if(!aresetn) begin
            have_aw<=0; have_w<=0; awaddr<=0; wdata<=0; wstrb<=0;
            s_axi_bvalid<=0; s_axi_bresp<=0;
            s_axi_rvalid<=0; s_axi_rresp<=0; s_axi_rdata<=0; soft_reset<=0;
        end else begin
            soft_reset<=0;
            if(s_axi_awvalid && s_axi_awready) begin have_aw<=1; awaddr<=s_axi_awaddr; end
            if(s_axi_wvalid && s_axi_wready) begin have_w<=1; wdata<=s_axi_wdata; wstrb<=s_axi_wstrb; end
            if(have_aw && have_w && !s_axi_bvalid) begin
                have_aw<=0; have_w<=0; s_axi_bvalid<=1;
                s_axi_bresp<=awaddr==8'h04 ? 2'b00 : 2'b10;
                if(awaddr==8'h04 && wstrb[0] && wdata[0]) soft_reset<=1;
            end
            if(s_axi_bvalid && s_axi_bready) s_axi_bvalid<=0;
            if(s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid<=1; s_axi_rresp<=0;
                case(s_axi_araddr)
                    8'h00: s_axi_rdata<=32'h52530003;
                    8'h04: s_axi_rdata<=0;
                    8'h08: s_axi_rdata<={30'd0,error,(busy || codec_busy || resident_busy)};
                    8'h0c: s_axi_rdata<=cycles;
                    8'h10: s_axi_rdata<=packets;
                    8'h14: s_axi_rdata<=32'h00030020 | ((FILTERS*P_CONTEXTS)<<24) | (COLS<<8);
                    8'h18: s_axi_rdata<=(FILTERS<<16) | (Q_CONTEXTS<<8) | P_CONTEXTS;
                    8'h20: s_axi_rdata<=stream_inputs;
                    8'h24: s_axi_rdata<=stream_outputs;
                    8'h28: s_axi_rdata<=mac_count;
                    8'h2c: s_axi_rdata<=load_wait;
                    8'h30: s_axi_rdata<=load_overlap;
                    8'h40: s_axi_rdata<=network_cycles;
                    8'h44: s_axi_rdata<=model_reads;
                    8'h48: s_axi_rdata<=act_reads;
                    8'h4c: s_axi_rdata<=act_write_bytes;
                    8'h50: s_axi_rdata<=images;
                    8'h54: s_axi_rdata<={model_valid,23'd0,current_layer};
                    default: begin s_axi_rdata<=0; s_axi_rresp<=2'b10; end
                endcase
            end
            if(s_axi_rvalid && s_axi_rready) s_axi_rvalid<=0;
        end
    end
endmodule
