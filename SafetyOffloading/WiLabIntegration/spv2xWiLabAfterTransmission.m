function stationManagement = spv2xWiLabAfterTransmission(stationManagement,timeManagement,txIDs,phyParams)
%SPV2XWILABAFTERTRANSMISSION Fill newly free WiLab buffers without polling.
s=stationManagement.spv2x; nowS=timeManagement.timeNow;
for v=txIDs(:)'
    if isempty(s.pending_tx{v}) || stationManagement.pckBuffer(v)>0 || s.current(v).valid
        continue
    end
    if timeManagement.timeNextPacket(v)<=nowS+phyParams.TTI+1e-10
        continue
    end
    item=s.pending_tx{v}(1); s.pending_tx{v}(1)=[];
    stationManagement.pckBuffer(v)=1; stationManagement.pckNextAttempt(v)=1;
    if isfield(stationManagement,'newDataIndicator'), stationManagement.newDataIndicator(v)=true; end
    s.current(v)=item;
end
stationManagement.spv2x=s;
end
