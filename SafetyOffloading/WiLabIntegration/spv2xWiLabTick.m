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
        idealCommunication=isfield(cfg.compute,'ideal_communication') && cfg.compute.ideal_communication;
        if s.tasks(tid).action=="local" || idealCommunication
            s.tasks(tid).result_delivered_ms=s.tasks(tid).cpu_finish_ms;
            if s.tasks(tid).result_delivered_ms<=s.tasks(tid).deadline_ms, s.tasks(tid).status="completed"; else, s.tasks(tid).status="late_cpu"; end
        else
            [s,items]=makeFragments(s,tid,v,s.tasks(tid).origin,s.tasks(tid).output_bytes,"result",nowMs);
            s.tasks(tid).output_fragments=numel(items); s.pending_tx{v}=[s.pending_tx{v};items];
        end
    end
end

% Release tasks at deterministic application epochs.
while s.release_cursor<=numel(s.tasks) && s.tasks(s.release_cursor).release_ms<=nowMs+1e-6
    tid=s.release_cursor; origin=s.tasks(tid).origin;
    candidates=helperCandidates(origin,ids,positionManagement,cfg.compute.helper_range_m,cfg.compute.max_helpers);
    [action,helper,risk,cbr]=chooseOverlayAction(s,origin,candidates,stationManagement,positionManagement);
    helperDistance=NaN; helperQueueLength=0;
    if helper>0
        helperDistance=positionManagement.distanceReal(origin,helper);
        helperQueueLength=numel(s.cpu_queue{helper})+(s.cpu_active(helper)>0);
    end
    originQueueLength=numel(s.cpu_queue{origin})+(s.cpu_active(origin)>0);
    helperRank=0;
    if helper>0, helperRank=find(candidates==helper,1); end
    s.decision_counter=s.decision_counter+1; s.tasks(tid).decision_id=s.decision_counter;
    s.tasks(tid).action=action; s.tasks(tid).helper=helper;
    s.decision_rows(end+1,:)={simParams.seed,s.decision_counter,nowMs,origin,tid,char(action),helper,helperRank, ...
        risk,cbr,numel(candidates),helperDistance,originQueueLength,helperQueueLength, ...
        cfg.compute.input_bytes,cfg.compute.output_bytes,cfg.compute.cycles_per_task, ...
        cfg.compute.deadline_ms,char(s.policy),char(s.world)};
    if action=="local"
        s.tasks(tid).cpu_owner=origin; s.tasks(tid).status="queued_local"; s.cpu_queue{origin}(end+1)=tid;
    elseif action=="offload"
        idealCommunication=isfield(cfg.compute,'ideal_communication') && cfg.compute.ideal_communication;
        s.tasks(tid).cpu_owner=helper;
        if idealCommunication
            s.tasks(tid).input_delivered_ms=nowMs;
            s.tasks(tid).status="queued_remote";
            s.cpu_queue{helper}(end+1)=tid;
        else
            [s,items]=makeFragments(s,tid,origin,helper,s.tasks(tid).input_bytes,"input",nowMs);
            s.tasks(tid).input_fragments=numel(items);
            s.tasks(tid).status="input_in_flight"; s.pending_tx{origin}=[s.pending_tx{origin};items];
        end
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
    if isempty(s.pending_tx{v}), continue; end
    if stationManagement.pckBuffer(v)>0 || s.current(v).valid, continue; end
    if timeManagement.timeNextPacket(v)<=nowS+phyParams.TTI+1e-10, continue; end
    item=s.pending_tx{v}(1); s.pending_tx{v}(1)=[];
    stationManagement.pckBuffer(v)=1; stationManagement.pckNextAttempt(v)=1;
    timeManagement.timeLastPacket(v)=nowS;
    if isfield(stationManagement,'newDataIndicator'), stationManagement.newDataIndicator(v)=true; end
    s.current(v)=item;
end

% The overlay is event driven.  Polling every PHY TTI made a 200 s run
% perform 400,000 unnecessary full-fleet scans.  The only autonomous
% overlay events are the next task release and the next CPU completion;
% packet-buffer handoff is triggered directly by a completed transmission.
nextRelease=inf;
if s.release_cursor<=numel(s.tasks)
    nextRelease=s.tasks(s.release_cursor).release_ms/1000;
end
nextCpu=min(s.cpu_finish_s);
s.next_tick_s=min(nextRelease,nextCpu);
stationManagement.spv2x=s;
end

function candidates=helperCandidates(origin,ids,positionManagement,maxRange,maxHelpers)
if ~isfield(positionManagement,'distanceReal'), candidates=[]; return; end
d=positionManagement.distanceReal(origin,ids); valid=ids(d>0 & d<=maxRange);
if isempty(valid), candidates=[]; return; end
[~,order]=sort(positionManagement.distanceReal(origin,valid)); valid=valid(order);
candidates=valid(1:min(maxHelpers,numel(valid)))';
end

function [action,helper,risk,cbr]=chooseOverlayAction(s,origin,candidates,stationManagement,positionManagement)
cfg=s.cfg; helper=0; cbr=0;
if isfield(stationManagement,'cbrCV2Xvalues')
    values=stationManagement.cbrCV2Xvalues(origin,:); values=values(values>=0);
    if ~isempty(values), cbr=values(end); end
end
risk=min(1,0.002+0.015*cbr+0.003*(cfg.compute.input_bytes/cfg.radio.max_transport_block_bytes));
if s.world=="baseline" || s.policy=="no_offload" || isempty(candidates)
    action="local"; return
end
switch s.policy
    case {"oracle_instant","oracle_actual","safety_oracle"}
        nowS=s.tasks(s.release_cursor).release_ms/1000;
        deadlineMs=s.tasks(s.release_cursor).deadline_ms;
        localFinishMs=forecastCpuFinishMs(s,origin,nowS,cfg.compute.cycles_per_task);
        helperFinishMs=inf(size(candidates));
        for ci=1:numel(candidates)
            helperFinishMs(ci)=forecastCpuFinishMs(s,candidates(ci),nowS,cfg.compute.cycles_per_task);
        end
        [bestHelperFinish,bestIndex]=min(helperFinishMs);
        remoteFinishMs=bestHelperFinish;
        if s.policy~="oracle_instant"
            fragments=ceil(cfg.compute.input_bytes/cfg.radio.max_transport_block_bytes)+ ...
                ceil(cfg.compute.output_bytes/cfg.radio.max_transport_block_bytes);
            remoteFinishMs=remoteFinishMs+fragments*cfg.radio.selection_window_ms;
        end
        safetyFeasible=s.policy~="safety_oracle" || ...
            risk<=cfg.control.risk_harm_limit;
        if s.policy=="safety_oracle" && isfield(cfg.compute,'oracle_max_offloads')
            admitted=sum(arrayfun(@(t)t.action=="offload",s.tasks));
            safetyFeasible=safetyFeasible && admitted<cfg.compute.oracle_max_offloads;
        end
        if localFinishMs<=deadlineMs
            action="local";
        elseif remoteFinishMs<=deadlineMs && safetyFeasible
            action="offload"; helper=candidates(bestIndex);
        else
            action="local";
        end
    case "dqn"
        originQueue=numel(s.cpu_queue{origin})+(s.cpu_active(origin)>0);
        helperDistance=0; helperQueue=0;
        if ~isempty(candidates)
            helperDistance=positionManagement.distanceReal(origin,candidates(1));
            helperQueue=numel(s.cpu_queue{candidates(1)})+(s.cpu_active(candidates(1))>0);
        end
        observation=[cbr,s.density,numel(candidates),helperDistance,originQueue,helperQueue, ...
            cfg.compute.input_bytes,cfg.compute.output_bytes,cfg.compute.cycles_per_task,cfg.compute.deadline_ms];
        actionIndex=scoreDqn(s.dqn,observation,numel(candidates));
        if actionIndex==0
            % Rejection is dominated by local execution whenever the local
            % queue can still meet this task's deadline.  Keep the learned
            % action inside the feasible set at deployment.
            queuedCycles=sum(arrayfun(@(id)s.tasks(id).cycles,s.cpu_queue{origin}));
            decisionNowS=s.tasks(s.release_cursor).release_ms/1000;
            if s.cpu_active(origin)>0
                queuedCycles=queuedCycles+max(0,s.cpu_finish_s(origin)-decisionNowS)*s.cpu_rate(origin);
            end
            localFinishMs=decisionNowS*1000+(queuedCycles+cfg.compute.cycles_per_task)/s.cpu_rate(origin)*1000;
            if localFinishMs<=s.tasks(s.release_cursor).deadline_ms
                action="local";
            else
                action="reject";
            end
        elseif actionIndex==1
            action="local";
        else
            action="offload"; helper=candidates(actionIndex-1);
        end
    case "dqn_explore"
        % Collect valid action-space experience without bypassing WiLab MAC.
        % 1=reject, 2=local, 3..N=ranked helper choices.
        u=stableUniform(s.scenario_id,"dqn-explore",s.decision_counter+1,origin);
        actionIndex=min(2+numel(candidates),floor(u*(2+numel(candidates)))+1);
        if actionIndex==1
            action="reject";
        elseif actionIndex==2
            action="local";
        else
            action="offload"; helper=candidates(actionIndex-2);
        end
    case "acceptance_split"
        if isempty(s.cpu_queue{origin}) && s.cpu_active(origin)==0
            action="local";
        else
            action="offload"; helper=candidates(1);
        end
    case "cbr_threshold"
        helper=candidates(1);
        if cbr<=cfg.control.cbr_threshold, action="offload"; else, action="local"; end
    case "risk_heuristic"
        helper=candidates(1);
        originQueue=numel(s.cpu_queue{origin})+(s.cpu_active(origin)>0);
        helperDistance=positionManagement.distanceReal(origin,helper);
        helperQueue=numel(s.cpu_queue{helper})+(s.cpu_active(helper)>0);
        observation=[cbr,s.density,numel(candidates),helperDistance,originQueue,helperQueue, ...
            cfg.compute.input_bytes,cfg.compute.output_bytes,cfg.compute.cycles_per_task,cfg.compute.deadline_ms];
        risk=scoreRisk(s.risk_model,observation);
        if cbr>=cfg.control.risk_ood_cbr || risk>=1
            action="reject";
        elseif risk+cfg.control.risk_margin<=cfg.control.risk_harm_limit
            action="offload";
        else
            action="local";
        end
    otherwise
        helper=candidates(1);
        action="offload";
end
end

function finishMs=forecastCpuFinishMs(s,vehicle,nowS,newCycles)
backlogCycles=sum(arrayfun(@(id)s.tasks(id).cycles,s.cpu_queue{vehicle}));
if s.cpu_active(vehicle)>0
    backlogCycles=backlogCycles+max(0,s.cpu_finish_s(vehicle)-nowS)*s.cpu_rate(vehicle);
end
finishMs=nowS*1000+(backlogCycles+newCycles)/s.cpu_rate(vehicle)*1000;
end

function upper=scoreRisk(model,observation)
means=double(model.means(:))'; scales=double(model.scales(:))';
x=(double(observation)-means)./scales; weights=double(model.weights(:))';
raw=1/(1+exp(-(weights(1)+sum(weights(2:end).*x)))); upper=1;
bins=model.calibration_bins;
for i=1:numel(bins)
    if raw>=bins(i).p_min && raw<=bins(i).p_max
        upper=max(raw,bins(i).empirical_upper); return
    end
end
end

function actionIndex=scoreDqn(model,observation,nCandidates)
means=double(model.means(:))'; scales=double(model.scales(:))';
x=((double(observation)-means)./scales)';
hidden=max(0,double(model.hidden_weights)*x+double(model.hidden_bias(:)));
q=double(model.output_weights)*hidden+double(model.output_bias(:));
nValid=min(numel(q),2+nCandidates); [~,index]=max(q(1:nValid)); actionIndex=index-1;
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

