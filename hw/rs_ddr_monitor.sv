// Passive counts of accepted 64-bit PS HP0 AXI beats, not DRAM row commands.
module rs_ddr_monitor (
    (* X_INTERFACE_PARAMETER="ASSOCIATED_BUSIF s_axi, ASSOCIATED_RESET aresetn, FREQ_HZ 100000000" *)
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
    input wire hp_rvalid,hp_rready,hp_wvalid,hp_wready,
    input wire [7:0] hp_wstrb,
    input wire hp_arvalid,hp_arready,hp_awvalid,hp_awready
);
    reg have_aw,have_w,clear;
    reg [7:0] awaddr;
    reg [31:0] wdata;
    reg [3:0] wstrb;
    reg [31:0] read_beats,write_beats,write_bytes,read_requests,write_requests;
    wire [3:0] accepted_bytes={3'd0,hp_wstrb[0]}+{3'd0,hp_wstrb[1]}+{3'd0,hp_wstrb[2]}+{3'd0,hp_wstrb[3]}+{3'd0,hp_wstrb[4]}+{3'd0,hp_wstrb[5]}+{3'd0,hp_wstrb[6]}+{3'd0,hp_wstrb[7]};
    assign s_axi_awready=aresetn && !have_aw && !s_axi_bvalid;
    assign s_axi_wready=aresetn && !have_w && !s_axi_bvalid;
    assign s_axi_arready=aresetn && !s_axi_rvalid;
    always @(posedge aclk) begin
        if(!aresetn || clear) begin
            read_beats<=0;write_beats<=0;write_bytes<=0;read_requests<=0;write_requests<=0;
        end else begin
            if(hp_rvalid && hp_rready) read_beats<=read_beats+1;
            if(hp_wvalid && hp_wready) begin write_beats<=write_beats+1;write_bytes<=write_bytes+accepted_bytes;end
            if(hp_arvalid && hp_arready) read_requests<=read_requests+1;
            if(hp_awvalid && hp_awready) write_requests<=write_requests+1;
        end
        if(!aresetn) begin
            have_aw<=0;have_w<=0;clear<=0;awaddr<=0;wdata<=0;wstrb<=0;
            s_axi_bvalid<=0;s_axi_bresp<=0;s_axi_rvalid<=0;s_axi_rresp<=0;s_axi_rdata<=0;
        end else begin
            clear<=0;
            if(s_axi_awvalid && s_axi_awready) begin have_aw<=1;awaddr<=s_axi_awaddr;end
            if(s_axi_wvalid && s_axi_wready) begin have_w<=1;wdata<=s_axi_wdata;wstrb<=s_axi_wstrb;end
            if(have_aw && have_w && !s_axi_bvalid) begin
                have_aw<=0;have_w<=0;s_axi_bvalid<=1;s_axi_bresp<=awaddr==4 ? 0 : 2;
                if(awaddr==4 && wstrb[0] && wdata[0]) clear<=1;
            end
            if(s_axi_bvalid && s_axi_bready) s_axi_bvalid<=0;
            if(s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid<=1;s_axi_rresp<=0;
                case(s_axi_araddr)
                0:s_axi_rdata<=32'h5253444d;
                4:s_axi_rdata<=0;
                8:s_axi_rdata<=read_beats;
                12:s_axi_rdata<=write_beats;
                16:s_axi_rdata<=write_bytes;
                20:s_axi_rdata<=read_requests;
                24:s_axi_rdata<=write_requests;
                default:begin s_axi_rdata<=0;s_axi_rresp<=2;end
                endcase
            end
            if(s_axi_rvalid && s_axi_rready) s_axi_rvalid<=0;
        end
    end
endmodule
