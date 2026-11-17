function [stationManagement,timeManagement] = spv2xWiLabTick(stationManagement,timeManagement,positionManagement,~,simParams,phyParams)
%SPV2XWILABTICK Advance tasks/CPUs and submit packets to WiLab buffers.
s = stationManagement.spv2x; cfg=s.cfg; nowS=timeManagement.timeNow; nowMs=nowS*1000;
ids=stationManagement.activeIDsCV2X(:);

% Complete processors and enqueue result fragments.
for v=ids'
    tid=s.cpu_active(v);
    if tid>0 && s.cpu_finish_s(v)<=nowS+1e-10
        s.tasks(tid).cpu_finish_ms=s.cpu_finish_s(v)*1000;
        s.cpu_active(v)=0; s.cpu_finish_s(v)=inf;
        if s.tasks(tid).action=="local"
            s.tasks(tid).result_delivered_ms=s.tasks(tid).cpu_finish_ms;
            if s.tasks(tid).result_delivered_ms<=s.tasks(tid).deadline_ms, s.tasks(tid).status="completed"; else, s.tasks(tid).status="late_cpu"; end
        else
            [s,items]=makeFragments(s,tid,v,s.tasks(tid).origin,s.tasks(tid).output_bytes,"result",nowMs);
            s.tasks(tid).output_fragments=numel(items); s.pending_tx{v}=[s.pending_tx{v};items];
        end
    end
end

% Release tasks at deterministic application epochs.
while s.release_cursor<=numel(s.tasks) && s.tasks(s.release_cursor).release_ms<=nowMs+1e-8
    tid=s.release_cursor; origin=s.tasks(tid).origin;
    candidates=helperCandidates(origin,ids,positionManagement,cfg.compute.helper_range_m,cfg.compute.max_helpers);
    [action,helper,risk,cbr]=chooseOverlayAction(s,origin,candidates,stationManagement);
    helperDistance=NaN; helperQueueLength=0;
    if helper>0
        helperDistance=positionManagement.distanceReal(origin,helper);
        helperQueueLength=numel(s.cpu_queue{helper})+(s.cpu_active(helper)>0);
    end
    originQueueLength=numel(s.cpu_queue{origin})+(s.cpu_active(origin)>0);
    s.decision_counter=s.decision_counter+1; s.tasks(tid).decision_id=s.decision_counter;
    s.tasks(tid).action=action; s.tasks(tid).helper=helper;
    s.decision_rows(end+1,:)={simParams.seed,s.decision_counter,nowMs,origin,tid,char(action),helper, ...
        risk,cbr,numel(candidates),helperDistance,originQueueLength,helperQueueLength, ...
        cfg.compute.input_bytes,cfg.compute.output_bytes,cfg.compute.cycles_per_task, ...
        cfg.compute.deadline_ms,char(s.policy),char(s.world)};
    if action=="local"
        s.tasks(tid).cpu_owner=origin; s.tasks(tid).status="queued_local"; s.cpu_queue{origin}(end+1)=tid;
    elseif action=="offload"
        [s,items]=makeFragments(s,tid,origin,helper,s.tasks(tid).input_bytes,"input",nowMs);
        s.tasks(tid).input_fragments=numel(items); s.tasks(tid).cpu_owner=helper;
        s.tasks(tid).status="input_in_flight"; s.pending_tx{origin}=[s.pending_tx{origin};items];
    else
        s.tasks(tid).status="rejected";
    end
    s.release_cursor=s.release_cursor+1;
end

% Start idle non-preemptive EDF processors.
for v=ids'
    if s.cpu_active(v)==0 && ~isempty(s.cpu_queue{v})
        q=s.cpu_queue{v}; deadlines=arrayfun(@(id)s.tasks(id).deadline_ms,q);
        [~,ix]=min(deadlines); tid=q(ix); q(ix)=[]; s.cpu_queue{v}=q;
        if nowMs<s.tasks(tid).deadline_ms
            s.cpu_active(v)=tid; s.tasks(tid).cpu_start_ms=nowMs;
            s.cpu_finish_s(v)=nowS+s.tasks(tid).cycles/s.cpu_rate(v);
        else
            s.tasks(tid).status="expired_cpu_queue";
        end
    end
end

% Submit one pending fragment per free vehicle. WiLab owns resource choice.
for v=ids'
    if isempty(s.pending_tx{v}) || stationManagement.pckBuffer(v)>0 || s.current(v).valid, continue; end
    if timeManagement.timeNextPacket(v)<=nowS+phyParams.TTI+1e-10, continue; end
    item=s.pending_tx{v}(1); s.pending_tx{v}(1)=[];
    stationManagement.pckBuffer(v)=1; stationManagement.pckNextAttempt(v)=1;
    timeManagement.timeLastPacket(v)=nowS;
    if isfield(stationManagement,'newDataIndicator'), stationManagement.newDataIndicator(v)=true; end
    s.current(v)=item;
end

s.next_tick_s=round(nowS+s.tick_s,10);
stationManagement.spv2x=s;
end

function candidates=helperCandidates(origin,ids,positionManagement,maxRange,maxHelpers)
if ~isfield(positionManagement,'distanceReal'), candidates=[]; return; end
d=positionManagement.distanceReal(origin,ids); valid=ids(d>0 & d<=maxRange);
if isempty(valid), candidates=[]; return; end
[~,order]=sort(positionManagement.distanceReal(origin,valid)); valid=valid(order);
candidates=valid(1:min(maxHelpers,numel(valid)))';
end

function [action,helper,risk,cbr]=chooseOverlayAction(s,origin,candidates,stationManagement)
cfg=s.cfg; helper=0; cbr=0;
if isfield(stationManagement,'cbrCV2Xvalues')
    values=stationManagement.cbrCV2Xvalues(origin,:); values=values(values>=0);
    if ~isempty(values), cbr=values(end); end
end
risk=min(1,0.002+0.015*cbr+0.003*(cfg.compute.input_bytes/cfg.radio.max_transport_block_bytes));
if s.world=="baseline" || s.policy=="no_offload" || isempty(candidates)
    action="local"; return
end
helper=candidates(1);
switch s.policy
    case "cbr_threshold"
        if cbr<=cfg.control.cbr_threshold, action="offload"; else, action="local"; end
    case "risk_heuristic"
        if cbr>=cfg.control.risk_ood_cbr
            action="reject";
        elseif risk+cfg.control.risk_margin<=cfg.control.risk_harm_limit
            action="offload";
        else
            action="local";
        end
    otherwise
        action="offload";
end
end

function [s,items]=makeFragments(s,tid,src,dst,bytes,stage,createdMs)
n=ceil(bytes/s.cfg.radio.max_transport_block_bytes); items=repmat(s.empty_meta,n,1);
for f=1:n
    s.packet_counter=s.packet_counter+1; m=s.empty_meta; m.valid=true; m.packet_id=s.packet_counter;
    m.kind="compute"; m.task_id=tid; m.stage=stage; m.fragment=f; m.fragments=n;
    m.src=src; m.dst=dst; m.created_ms=createdMs; m.deadline_ms=s.tasks(tid).deadline_ms;
    m.seq=f; m.flow_id=1000000+tid; m.bytes=min(s.cfg.radio.max_transport_block_bytes,bytes-(f-1)*s.cfg.radio.max_transport_block_bytes);
    items(f)=m;
end
end

