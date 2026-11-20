function [stationManagement,timeManagement] = spv2xWiLabAfterTransmission(stationManagement,timeManagement,txIDs,phyParams)
%SPV2XWILABAFTERTRANSMISSION Fill newly free WiLab buffers without polling.
s=stationManagement.spv2x; nowS=timeManagement.timeNow;
for v=txIDs(:)'
    % Reception accounting has already consumed the completed transmission.
    % Clear its tag here as well so packet-buffer refill does not depend on
    % the ordering of the KPI and MAC bookkeeping callbacks.
    if s.current(v).valid && ...
            stationManagement.pckTxOccurring(v)>=stationManagement.cv2xNumberOfReplicas(v)
        s.current(v)=s.empty_meta;
    end
    if isempty(s.pending_tx{v}) || stationManagement.pckBuffer(v)>0 || s.current(v).valid
        continue
    end
    if timeManagement.timeNextPacket(v)<=nowS+phyParams.TTI+1e-10
        continue
    end
    item=s.pending_tx{v}(1); s.pending_tx{v}(1)=[];
    stationManagement.pckBuffer(v)=1; stationManagement.pckNextAttempt(v)=1;
    % This is a new packet, not a HARQ repetition of the packet that just
    % left the buffer.  Timestamp it so WiLab's Mode-2 allocator selects a
    % fresh resource inside the selection window instead of waiting for the
    % transmitter's next 100 ms SPS reservation.
    timeManagement.timeLastPacket(v)=nowS;
    if isfield(stationManagement,'newDataIndicator'), stationManagement.newDataIndicator(v)=true; end
    s.current(v)=item;
end
stationManagement.spv2x=s;
end
