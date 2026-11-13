// Resident model/tensor BRAM and hardware graph/tile sequencer. Legacy tile
// commands pass through unchanged. Opcode 4 uploads a model once; opcode 5
// accepts one packed image and returns final logits (or a debug layer prefix).
module rs_resident #(
    parameter integer COLS=8, FILTERS=2, P_CONTEXTS=1,
    parameter integer MODEL_WORDS=24576, ACT_WORDS=12288
) (
    input wire clk,resetn,
    input wire [31:0] s_data,input wire s_valid,output wire s_ready,
    output reg [31:0] c_data,output reg c_valid,input wire c_ready,
    input wire [31:0] e_data,input wire e_valid,e_last,output wire e_ready,
    output reg [31:0] m_data,output reg m_valid,m_last,input wire m_ready,
    input wire core_error,
    output wire busy,output reg error,output reg model_valid,
    output reg [31:0] network_cycles,model_reads,activation_reads,activation_write_bytes,
    output reg [31:0] completed_images,
    output wire [7:0] current_layer
);
    localparam MAGIC=32'h52534e32, NFMAX=FILTERS*P_CONTEXTS;
    localparam H=0,DISPATCH=1,LEGACY_H=2,LEGACY_DATA=3,MODEL_LOAD=4,MODEL_CHECK=5,
       ACK=6,IMAGE_LOAD=7,DESC_REQ=8,DESC_WAIT=9,DESC_CAP=10,LAYER_SETUP=11,
       TILE_SETUP=12,PARAM_INIT=13,PARAM_REQ=14,PARAM_WAIT=15,PARAM_CAP=16,
       START_HEADER=17,SEND_HEADER=18,EXTRA_PARAM=19,PAYLOAD=20,
       ACT_REQ=21,ACT_WAIT=22,WEIGHT_REQ=23,WEIGHT_WAIT=24,TILE_WAIT=25,
       TILE_DONE=26,LAYER_DONE=27,EXPORT_REQ=28,EXPORT_WAIT=29,EXPORT_WORD=30,FAULT=31,BASE_INIT=32,LAYER_PREP=33;
    reg [5:0] state;
    reg [31:0] h[0:15],mh[0:15],d[0:15];
    reg [3:0] hi,di;
    reg [7:0] layer,layer_limit,tf,ty,tx,nf,nr,nw,pw,ph;
    reg new_filter;
    reg [7:0] ch,iy,ix,wg,wr,wt,out_g,out_y,out_x;
    reg [7:0] log_w,log_plane,log_c,log_ow,log_oplane;
    reg [31:0] position,upload_index,export_index,export_count,req_tag;
    reg [31:0] param_index,param_count;
    reg [31:0] quant[0:3*NFMAX-1];
    reg [2:0] payload_phase;
    reg [7:0] fc_class;
    reg [31:0] fc_index,payload_last;
    reg [23:0] pool_count;
    reg [16:0] filter_start,next_filter_start,base_pointer,weight_pointer;
    reg [16:0] weight_bases[0:NFMAX-1];
    reg [9:0] filter_bytes;
    reg [7:0] base_index;
    reg run_active;
    reg a_cache_valid,w_cache_valid;
    reg [31:0] a_cache_tag,w_cache_tag,a_cache,w_cache;
    reg [31:0] read_word_address;
    wire legacy=state==LEGACY_H || state==LEGACY_DATA;
    wire resident_running=run_active;
    assign busy=state!=H || hi!=0;
    assign current_layer=layer;
    wire [7:0] op=d[0][7:0];
    wire [7:0] kernel=d[10][7:0],stride=d[10][15:8],pad=d[10][23:16];
    wire [7:0] next_nf=d[7]-tf<NFMAX ? d[7]-tf : NFMAX;
    wire [7:0] next_nr=d[8]-ty<COLS ? d[8]-ty : COLS;
    wire [7:0] width_limit=stride==2 ? ((34-kernel)>>1)+1 : 34-kernel+1;
    wire [7:0] bounded_width=width_limit<32 ? width_limit : 8'd32;
    wire [7:0] next_nw=d[9]-tx<bounded_width ? d[9]-tx : bounded_width;
    wire signed [16:0] sy=$signed({1'b0,(stride==2 ? {ty,1'b0} : {1'b0,ty})})+$signed({1'b0,iy})-$signed({1'b0,pad});
    wire signed [16:0] sx=$signed({1'b0,(stride==2 ? {tx,1'b0} : {1'b0,tx})})+$signed({1'b0,ix})-$signed({1'b0,pad});
    wire padding=sy<0 || sx<0 || sy>=$signed(d[5]) || sx>=$signed(d[6]);
    wire [16:0] weight_address=weight_pointer;
    // The resident ABI has a 48 KiB tensor address space. Its compiler checks
    // this fixed graph's shapes/liveness before creating descriptors.
    wire [15:0] input_address=d[1][15:0] | (({8'd0,ch}<<log_plane) | (sy[15:0]<<log_w) | sx[15:0]);
    wire [31:0] conv_output_address=d[3] | (((tf+out_g)<<log_oplane) | ((ty+out_y)<<log_ow) | (tx+out_x));
    reg [31:0] payload_address,payload_value;
    reg payload_is_model,payload_is_word,payload_padding,payload_available;
    reg [31:0] request_address;
    reg request_valid,request_last,issued_all;
    reg request_is_model,request_is_word,request_padding;
    wire [31:0] payload_tag=request_address>>2;
    wire c_fire=c_valid && c_ready;
    wire issue_payload=state==PAYLOAD && !issued_all && (!request_valid || c_fire);
    wire last_generated=op==0 ? payload_phase==1 && wt==kernel-1 && wr==kernel-1 && wg==nf-1 && ch==d[4]-1 :
                        op==1 || op==2 ? fc_index==payload_last :
                        payload_phase==2 && fc_index==payload_last && fc_class==d[7]-1;
    wire e_fire=e_valid && e_ready;
    wire s_fire=s_valid && s_ready;
    assign s_ready=resetn && (state==H || state==MODEL_LOAD || state==IMAGE_LOAD ||
                    (state==LEGACY_DATA && c_ready));
    assign e_ready=resetn && (legacy ? m_ready : resident_running);

    function [7:0] shape_shift(input [31:0] value);
        begin
            case(value)
            1:shape_shift=0;2:shape_shift=1;4:shape_shift=2;8:shape_shift=3;
            16:shape_shift=4;32:shape_shift=5;64:shape_shift=6;128:shape_shift=7;
            256:shape_shift=8;512:shape_shift=9;1024:shape_shift=10;
            default:shape_shift=0;
            endcase
        end
    endfunction

    // One synchronous read and one write port per model memory; four byte banks
    // allow masked tensor writes without an activation read/modify/write cycle.
    (* ram_style="block" *) reg [31:0] model[0:MODEL_WORDS-1];
    wire model_rd_en=state==DESC_REQ || state==PARAM_REQ || state==WEIGHT_REQ;
    wire [31:0] model_rd_addr=state==DESC_REQ ? mh[8]+{layer,4'b0}+di :
                state==PARAM_REQ ? d[12]+(op==0 ? tf*3 : 0)+param_index : req_tag;
    reg [31:0] model_rd;
    wire act_rd_en=state==ACT_REQ || state==EXPORT_REQ;
    wire [31:0] act_rd_addr=state==EXPORT_REQ ? (d[3]>>2)+export_index : req_tag;
    wire [31:0] act_rd;
    reg [31:0] act_write_address,act_write_data;
    reg [3:0] act_write_mask;
    always @* begin
        act_write_address=0;act_write_data=0;act_write_mask=0;
        if(state==IMAGE_LOAD && s_fire) begin
            act_write_address=upload_index;act_write_data=s_data;act_write_mask=15;
        end else if(resident_running && e_fire) begin
            if(op==0) begin
                act_write_address=conv_output_address>>2;
                act_write_data={4{e_data[7:0]}};act_write_mask=4'b0001<<conv_output_address[1:0];
            end else if(op==1) begin
                act_write_address=(d[3]+position)>>2;
                act_write_data={4{e_data[7:0]}};act_write_mask=4'b0001<<position[1:0];
            end else begin
                act_write_address=(d[3]>>2)+position;act_write_data=e_data;act_write_mask=15;
            end
        end
    end
    genvar lane;
    generate for(lane=0;lane<4;lane=lane+1) begin: activation_banks
        (* ram_style="block" *) reg [7:0] mem[0:ACT_WORDS-1];
        reg [7:0] rd;
        always @(posedge clk) begin
            if(act_rd_en && act_rd_addr<ACT_WORDS) rd<=mem[act_rd_addr];
            if(act_write_mask[lane] && act_write_address<ACT_WORDS) mem[act_write_address]<=act_write_data[8*lane+:8];
        end
        assign act_rd[8*lane+:8]=rd;
    end endgenerate
    always @(posedge clk) begin
        if(model_rd_en && model_rd_addr<MODEL_WORDS) model_rd<=model[model_rd_addr];
        if(state==MODEL_LOAD && s_fire) model[upload_index]<=s_data;
    end

    // The descriptor's image dimensions are powers of two. Row/plane address
    // shifts avoid serial generic multipliers in the request critical path.
    always @* begin
        payload_address=0;payload_is_model=0;payload_is_word=0;payload_padding=0;
        if(op==0) begin
            if(payload_phase==0) begin payload_address=input_address;payload_padding=padding;end
            else begin payload_address=weight_address;payload_is_model=1;end
        end else if(op==1) payload_address=(fc_index[0] ? d[2] : d[1])+(fc_index>>1);
        else if(op==2) payload_address=d[1]+fc_index;
        else if(payload_phase==0) begin payload_address=d[1]+(fc_index<<2);payload_is_word=1;end
        else if(payload_phase==1) begin payload_address=(d[12]+fc_class)<<2;payload_is_word=1;payload_is_model=1;end
        else begin payload_address=weight_pointer;payload_is_model=1;end
        payload_available=request_padding || (request_is_model ? w_cache_valid && w_cache_tag==payload_tag : a_cache_valid && a_cache_tag==payload_tag);
        payload_value=request_is_model ? w_cache : a_cache;
        if(!request_is_word) begin
            case(request_address[1:0])
            0:payload_value={{24{payload_value[7]}},payload_value[7:0]};
            1:payload_value={{24{payload_value[15]}},payload_value[15:8]};
            2:payload_value={{24{payload_value[23]}},payload_value[23:16]};
            3:payload_value={{24{payload_value[31]}},payload_value[31:24]};
            endcase
        end
        if(request_padding) payload_value=0;
    end
    always @* begin
        c_data=0;c_valid=0;m_data=0;m_valid=0;m_last=0;
        if(state==LEGACY_H) begin c_data=h[hi];c_valid=resetn;end
        if(state==LEGACY_DATA) begin c_data=s_data;c_valid=s_valid && resetn;end
        if(legacy) begin m_data=e_data;m_valid=e_valid;m_last=e_last;end
        if(state==ACK) begin m_data=MAGIC;m_valid=resetn;m_last=1;end
        if(state==EXPORT_WORD) begin m_data=act_rd;m_valid=resetn;m_last=export_index==export_count-1;end
        if(state==SEND_HEADER) begin
            c_valid=resetn;
            case(hi)
            0:c_data=op;
            1:c_data=op==0 ? kernel : op==2 ? d[5] : d[4];
            2:c_data=op==0 ? stride : op==2 ? d[4] : op==3 ? d[7] : 0;
            3:c_data=op==0 ? pw : 0;
            4:c_data=op==0 ? nw : 0;
            5:c_data=op==0 ? nr : 0;
            6:c_data=op==0 ? nf : 0;
            7:c_data=op==0 ? d[4] : 0;
            8:c_data={31'd0,d[0][8]};
            9:c_data=op==0 ? quant[0] : 0;
            10:c_data=op==0 ? quant[1] : op==1 ? quant[0] : 0;
            11:c_data=op==0 ? quant[2] : op==1 ? quant[2] : 0;
            12:c_data=op==0 && nf>1 ? quant[3] : 0;
            13:c_data=op==0 && nf>1 ? quant[4] : op==1 ? quant[1] : 0;
            14:c_data=op==0 && nf>1 ? quant[5] : 0;
            default:c_data=0;
            endcase
        end
        if(state==EXTRA_PARAM) begin c_data=quant[param_index];c_valid=resetn;end
        if(state==PAYLOAD) begin c_data=payload_value;c_valid=request_valid && payload_available && resetn;end
    end

    integer z;
    always @(posedge clk) begin
        if(!resetn) begin
            state<=H;hi<=0;di<=0;layer<=0;layer_limit<=0;error<=0;model_valid<=0;
            network_cycles<=0;model_reads<=0;activation_reads<=0;activation_write_bytes<=0;completed_images<=0;
            tf<=0;ty<=0;tx<=0;nf<=0;nr<=0;nw<=0;pw<=0;ph<=0;new_filter<=0;
            ch<=0;iy<=0;ix<=0;wg<=0;wr<=0;wt<=0;out_g<=0;out_y<=0;out_x<=0;
            log_w<=0;log_plane<=0;log_c<=0;log_ow<=0;log_oplane<=0;
            position<=0;upload_index<=0;export_index<=0;export_count<=0;req_tag<=0;
            param_index<=0;param_count<=0;payload_phase<=0;fc_class<=0;fc_index<=0;run_active<=0;
            a_cache_valid<=0;w_cache_valid<=0;a_cache_tag<=0;w_cache_tag<=0;a_cache<=0;w_cache<=0;
            request_address<=0;request_valid<=0;request_last<=0;issued_all<=0;
            request_is_model<=0;request_is_word<=0;request_padding<=0;
            pool_count<=0;payload_last<=0;
            filter_start<=0;next_filter_start<=0;base_pointer<=0;weight_pointer<=0;filter_bytes<=0;base_index<=0;
            for(z=0;z<NFMAX;z=z+1) weight_bases[z]<=0;
            for(z=0;z<16;z=z+1) begin h[z]<=0;mh[z]<=0;d[z]<=0;end
            for(z=0;z<3*NFMAX;z=z+1) quant[z]<=0;
        end else begin
            if(run_active) network_cycles<=network_cycles+1;
            if(model_rd_en) model_reads<=model_reads+1;
            if(act_rd_en) activation_reads<=activation_reads+1;
            if(act_write_mask!=0) activation_write_bytes<=activation_write_bytes+(act_write_mask==15 ? 4 : 1);
            if(resident_running && e_fire) begin
                position<=position+1;
                if(op==0) begin
                    if(out_x==nw-1) begin
                        out_x<=0;
                        if(out_y==nr-1) begin out_y<=0;out_g<=out_g+1;end
                        else out_y<=out_y+1;
                    end else out_x<=out_x+1;
                end
                if(e_last) state<=TILE_DONE;
            end
            case(state)
            H: if(s_fire) begin h[hi]<=s_data;if(hi==15) begin hi<=0;state<=DISPATCH;end else hi<=hi+1;end
            DISPATCH: begin
                if(h[0]<4) begin state<=LEGACY_H;hi<=0;end
                else if(h[0]==4 && h[1]>=16 && h[1]<=MODEL_WORDS) begin model_valid<=0;upload_index<=0;state<=MODEL_LOAD;end
                else if(h[0]==5 && model_valid && h[1]==mh[3] && h[15][0] && h[2]<=mh[2]) begin
                    upload_index<=0;layer<=0;layer_limit<=h[2]==0 ? mh[2][7:0] : h[2][7:0];
                    run_active<=1;a_cache_valid<=0;w_cache_valid<=0;state<=IMAGE_LOAD;
                end else begin state<=FAULT;error<=1;end
            end
            LEGACY_H: if(c_fire) begin if(hi==15) begin hi<=0;state<=LEGACY_DATA;end else hi<=hi+1;end
            LEGACY_DATA: if(e_fire && e_last) begin state<=H;hi<=0;end
            MODEL_LOAD: if(s_fire) begin
                if(upload_index<16) mh[upload_index]<=s_data;
                if(upload_index==h[1]-1) state<=MODEL_CHECK;
                else upload_index<=upload_index+1;
            end
            MODEL_CHECK: begin
                if(mh[0]==MAGIC && mh[1]==1 && mh[2]>0 && mh[2]<=32 && mh[3]==3072 &&
                   mh[4]==0 && mh[5]==h[1] && mh[5]>=16+(mh[2]<<4) && mh[8]==16 && mh[9]==16384) begin model_valid<=1;state<=ACK;end
                else begin state<=FAULT;error<=1;end
            end
            ACK: if(m_ready) state<=H;
            IMAGE_LOAD: if(s_fire) begin
                if(upload_index==(mh[3]>>2)-1) begin di<=0;state<=DESC_REQ;end
                else upload_index<=upload_index+1;
            end
            DESC_REQ: state<=DESC_WAIT;
            DESC_WAIT: state<=DESC_CAP;
            DESC_CAP: begin
                d[di]<=model_rd;
                if(di==15) begin di<=0;state<=LAYER_SETUP;end
                else begin di<=di+1;state<=DESC_REQ;end
            end
            LAYER_SETUP: begin
                tf<=0;ty<=0;tx<=0;new_filter<=1;position<=0;
                pool_count<=d[4][7:0]*d[5][15:0];
                payload_last<=op==1 ? (d[4]<<1)-1 : d[4]-1;
                filter_start<=d[11][16:0];
                filter_bytes<=kernel==3 ? ({2'd0,d[4][7:0]}<<3)+d[4][7:0] : d[4][7:0];
                a_cache_valid<=0;w_cache_valid<=0;
                log_w<=shape_shift(d[6]);log_plane<=shape_shift(d[5])+shape_shift(d[6]);
                log_c<=shape_shift(d[4]);log_ow<=shape_shift(d[9]);log_oplane<=shape_shift(d[8])+shape_shift(d[9]);
                if(op>3 || d[13]==0 || d[3]+(d[14][0] ? d[13]<<2 : d[13])>ACT_WORDS*4) begin state<=FAULT;error<=1;end
                else if(op==0) state<=TILE_SETUP;
                else if(op==1) begin param_count<=3;state<=PARAM_INIT;end
                else state<=op==2 ? LAYER_PREP : START_HEADER;
            end
            LAYER_PREP: begin payload_last<=pool_count-1'b1;state<=START_HEADER;end
            TILE_SETUP: begin
                nf<=next_nf;nr<=next_nr;nw<=next_nw;
                pw<=(stride==2 ? ((next_nw-1)<<1) : next_nw-1)+kernel;
                ph<=(stride==2 ? ((next_nr-1)<<1) : next_nr-1)+kernel;
                param_count<=3*next_nf;base_index<=0;base_pointer<=filter_start;state<=BASE_INIT;
            end
            BASE_INIT: begin
                weight_bases[base_index]<=base_pointer;
                base_pointer<=base_pointer+filter_bytes;
                if(base_index==nf-1) begin
                    next_filter_start<=base_pointer+filter_bytes;
                    state<=new_filter ? PARAM_INIT : START_HEADER;
                end else base_index<=base_index+1;
            end
            PARAM_INIT: begin param_index<=0;state<=PARAM_REQ;end
            PARAM_REQ: state<=PARAM_WAIT;
            PARAM_WAIT: state<=PARAM_CAP;
            PARAM_CAP: begin
                quant[param_index]<=model_rd;
                if(param_index==param_count-1) begin new_filter<=0;state<=START_HEADER;end
                else begin param_index<=param_index+1;state<=PARAM_REQ;end
            end
            START_HEADER: begin
                request_valid<=0;request_last<=0;issued_all<=0;
                hi<=0;ch<=0;iy<=0;ix<=0;wg<=0;wr<=0;wt<=0;
                out_g<=0;out_y<=0;out_x<=0;position<=0;
                payload_phase<=0;fc_class<=0;fc_index<=0;state<=SEND_HEADER;
                weight_pointer<=op==0 ? weight_bases[0] : d[11][16:0];
            end
            SEND_HEADER: if(c_fire) begin
                if(hi==15) begin hi<=0;if(op==0 && nf>2) begin param_index<=6;state<=EXTRA_PARAM;end else state<=PAYLOAD;end
                else hi<=hi+1;
            end
            EXTRA_PARAM: if(c_fire) begin
                if(param_index==3*nf-1) state<=PAYLOAD;
                else param_index<=param_index+1;
            end
            PAYLOAD: begin
                if(c_fire) begin
                    request_valid<=0;
                    if(request_last) state<=TILE_WAIT;
                end
                if(request_valid && !payload_available) begin
                    if((request_is_model && payload_tag>=MODEL_WORDS) || (!request_is_model && payload_tag>=ACT_WORDS)) begin state<=FAULT;error<=1;end
                    else begin req_tag<=payload_tag;state<=request_is_model ? WEIGHT_REQ : ACT_REQ;end
                end
                if(issue_payload) begin
                    request_address<=payload_address;request_is_model<=payload_is_model;
                    request_is_word<=payload_is_word;request_padding<=payload_padding;
                    request_valid<=1;request_last<=last_generated;
                    if(last_generated) issued_all<=1;
                    if((op==0 && payload_phase==1) || (op==3 && payload_phase==2)) weight_pointer<=weight_pointer+1'b1;
                    if(op==0) begin
                        if(payload_phase==0) begin
                            if(ix==pw-1) begin ix<=0;if(iy==ph-1) begin iy<=0;payload_phase<=1;end else iy<=iy+1;end
                            else ix<=ix+1;
                        end else if(wt==kernel-1) begin
                            wt<=0;
                            if(wr==kernel-1) begin
                                wr<=0;
                                if(wg==nf-1) begin
                                    wg<=0;payload_phase<=0;
                                    if(ch==d[4]-1) begin end
                                    else begin
                                        ch<=ch+1;
                                        for(z=0;z<NFMAX;z=z+1) weight_bases[z]<=weight_bases[z]+(kernel==3 ? 17'd9 : 17'd1);
                                        weight_pointer<=weight_bases[0]+(kernel==3 ? 17'd9 : 17'd1);
                                    end
                                end else begin wg<=wg+1;weight_pointer<=weight_bases[wg+1];end
                            end else wr<=wr+1;
                        end else wt<=wt+1;
                    end else if(op==1) begin if(fc_index==payload_last) begin end else fc_index<=fc_index+1;end
                    else if(op==2) begin if(fc_index==payload_last) begin end else fc_index<=fc_index+1;end
                    else if(payload_phase==0) begin
                        if(fc_index==payload_last) begin fc_index<=0;payload_phase<=1;end else fc_index<=fc_index+1;
                    end else if(payload_phase==1) begin fc_index<=0;payload_phase<=2;end
                    else if(fc_index==payload_last) begin
                        fc_index<=0;payload_phase<=1;
                        if(fc_class==d[7]-1) begin end else fc_class<=fc_class+1;
                    end else fc_index<=fc_index+1;
                end
            end
            ACT_REQ: state<=ACT_WAIT;
            ACT_WAIT: begin a_cache<=act_rd;a_cache_tag<=req_tag;a_cache_valid<=1;state<=PAYLOAD;end
            WEIGHT_REQ: state<=WEIGHT_WAIT;
            WEIGHT_WAIT: begin w_cache<=model_rd;w_cache_tag<=req_tag;w_cache_valid<=1;state<=PAYLOAD;end
            TILE_WAIT: begin end
            TILE_DONE: begin
                if(op!=0) state<=LAYER_DONE;
                else if(tx+nw<d[9]) begin tx<=tx+nw;state<=TILE_SETUP;end
                else if(ty+nr<d[8]) begin tx<=0;ty<=ty+nr;state<=TILE_SETUP;end
                else if(tf+nf<d[7]) begin tx<=0;ty<=0;tf<=tf+nf;filter_start<=next_filter_start;new_filter<=1;state<=TILE_SETUP;end
                else state<=LAYER_DONE;
            end
            LAYER_DONE: begin
                if(layer==layer_limit-1) begin
                    export_index<=0;export_count<=d[14][0] ? d[13] : (d[13]+3)>>2;state<=EXPORT_REQ;
                end else begin layer<=layer+1;di<=0;state<=DESC_REQ;end
            end
            EXPORT_REQ: state<=EXPORT_WAIT;
            EXPORT_WAIT: state<=EXPORT_WORD;
            EXPORT_WORD: if(m_ready) begin
                if(export_index==export_count-1) begin state<=H;hi<=0;run_active<=0;completed_images<=completed_images+1;end
                else begin export_index<=export_index+1;state<=EXPORT_REQ;end
            end
            default: begin end
            endcase
            if(core_error) begin state<=FAULT;error<=1;run_active<=0;end
        end
    end
endmodule
