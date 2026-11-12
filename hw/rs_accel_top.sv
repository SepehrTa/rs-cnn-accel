// AXI4-Lite control + AXIS data plane. Single clock domain, PS FCLK0 at 100MHz.
module rs_accel_top #(
    parameter integer COLS=8,
    parameter integer BANK_ADDR_BITS=8
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
    wire busy,error;
    wire [31:0] cycles,packets,buffer_data;
    wire buffer_valid,buffer_ready;
    rs_axis_buffer #(.BANK_ADDR_BITS(BANK_ADDR_BITS)) ingress(.clk(aclk),.resetn(core_resetn),
        .s_data(s_axis_tdata),.s_valid(s_axis_tvalid),.s_ready(s_axis_tready),
        .m_data(buffer_data),.m_valid(buffer_valid),.m_ready(buffer_ready));
    rs_stream_core #(.COLS(COLS)) core(.aclk(aclk),.aresetn(core_resetn),
        .s_axis_tdata(buffer_data),.s_axis_tvalid(buffer_valid),.s_axis_tready(buffer_ready),.s_axis_tlast(1'b0),
        .m_axis_tdata(m_axis_tdata),.m_axis_tvalid(m_axis_tvalid),.m_axis_tready(m_axis_tready),.m_axis_tlast(m_axis_tlast),
        .busy(busy),.error(error),.cycles(cycles),.packets(packets));
    assign m_axis_tkeep=4'hf;
    assign s_axi_awready=aresetn && !have_aw && !s_axi_bvalid;
    assign s_axi_wready=aresetn && !have_w && !s_axi_bvalid;
    assign s_axi_arready=aresetn && !s_axi_rvalid;
    always @(posedge aclk) begin
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
                    8'h00: s_axi_rdata<=32'h52530001;
                    8'h04: s_axi_rdata<=0;
                    8'h08: s_axi_rdata<={30'd0,error,busy};
                    8'h0c: s_axi_rdata<=cycles;
                    8'h10: s_axi_rdata<=packets;
                    8'h14: s_axi_rdata<=32'h02030020 | (COLS<<8);
                    default: begin s_axi_rdata<=0; s_axi_rresp<=2'b10; end
                endcase
            end
            if(s_axi_rvalid && s_axi_rready) s_axi_rvalid<=0;
        end
    end
endmodule
