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
for i=1:numel(rx)
    range=positionManagement.distanceReal(tx,rx(i));
    bin=find(range<=cfg.safety.distance_bins_m,1)-1; if isempty(bin), bin=numel(cfg.safety.distance_bins_m)-1; end
    s.safety_rows(end+1,:)=[tx seq tx rx(i) nowMs nowMs+cfg.safety.deadline_ms NaN 0 ...
        cfg.safety.packet_bytes range bin packetId];
end
if stationManagement.pckBuffer(tx)==0 && ~s.current(tx).valid
    m=s.empty_meta; m.valid=true; m.packet_id=packetId; m.kind="safety"; m.src=tx;
    m.created_ms=nowMs; m.deadline_ms=nowMs+cfg.safety.deadline_ms; m.seq=seq;
    m.flow_id=tx; m.bytes=appParams.beaconSizeBytes; s.current(tx)=m;
end
stationManagement.spv2x=s;
end

