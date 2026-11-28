function stationManagement = spv2xWiLabSafetyGenerated(stationManagement,positionManagement,timeManagement,tx,appParams)
%SPV2XWILABSAFETYGENERATED Register stable safety exposures before enqueue.
s=stationManagement.spv2x; cfg=s.cfg; nowMs=timeManagement.timeNow*1000;
s.packet_counter=s.packet_counter+1; s.safety_seq(tx)=s.safety_seq(tx)+1;
seq=s.safety_seq(tx); packetId=s.packet_counter;
if isfield(positionManagement,'distanceReal')
    d=positionManagement.distanceReal(tx,stationManagement.activeIDsCV2X);
    rx=stationManagement.activeIDsCV2X(d>0 & d<=s.evaluation_range_m);
else
    rx=[];
end
if s.receiver_sample_fraction<1 && ~isempty(rx)
    % Vectorized deterministic sampling, identical in every paired world.
    key=mod(double(tx)*73856093 + double(seq)*19349663 + ...
        double(rx)*83492791 + double(s.scenario_id)*2654435761,2^32)/2^32;
    keep=key<=s.receiver_sample_fraction;
    rx=rx(keep);
end
if ~isempty(rx)
    ranges=positionManagement.distanceReal(tx,rx(:));
    bins=zeros(numel(rx),1);
    for i=1:numel(rx)
        bin=find(ranges(i)<=cfg.safety.distance_bins_m,1)-1;
        if isempty(bin), bin=numel(cfg.safety.distance_bins_m)-1; end
        bins(i)=bin;
    end
    first=s.safety_row_cursor+1; last=first+numel(rx)-1;
    if last>size(s.safety_rows,1)
        newCapacity=max(last,ceil(size(s.safety_rows,1)*1.5));
        s.safety_rows(newCapacity,12)=0;
    end
    block=[repmat([tx seq tx],numel(rx),1),rx(:), ...
        repmat([nowMs nowMs+cfg.safety.deadline_ms NaN 0 cfg.safety.packet_bytes],numel(rx),1), ...
        ranges(:),bins,repmat(packetId,numel(rx),1)];
    s.safety_rows(first:last,:)=block; s.safety_row_cursor=last;
    s.safety_row_start(packetId)=first;
    s.safety_row_count(packetId)=numel(rx);
end
m=s.empty_meta; m.valid=true; m.packet_id=packetId; m.kind="safety"; m.src=tx;
m.created_ms=nowMs; m.deadline_ms=nowMs+cfg.safety.deadline_ms; m.seq=seq;
m.flow_id=tx; m.bytes=appParams.beaconSizeBytes;
if ~s.current(tx).valid
    s.current(tx)=m;
else
    % WiLab can buffer more than one safety packet.  Keep packet metadata in
    % the same FIFO order instead of silently losing the tag for queued data.
    s.meta_queue{tx}=[s.meta_queue{tx};m];
end
stationManagement.spv2x=s;
end

