function [stationManagement,timeManagement] = spv2xWiLabTick(stationManagement,timeManagement,positionManagement,~,simParams,phyParams)
%SPV2XWILABTICK Advance tasks/CPUs and submit packets to WiLab buffers.
s = stationManagement.spv2x;
if startsWith(string(s.policy),"workflow_")
    [stationManagement,timeManagement]=spv2xWorkflowBridge('tick', ...
        stationManagement,timeManagement,positionManagement,simParams,phyParams);
    return
end
cfg=s.cfg; nowS=timeManagement.timeNow; nowMs=nowS*1000;
ids=stationManagement.activeIDsCV2X(:);
if ~isfield(s,'offload_count'), s.offload_count=0; s.governor_spent=0; end
if ~isfield(s,'use_joint_scheduler'), s.use_joint_scheduler=false; end
if ~isfield(s,'pending_task_ids'), s.pending_task_ids=[]; end
cpuBecameFree=false;

% Complete processors and enqueue result fragments.
for v=ids'
    tid=s.cpu_active(v);
    if tid>0 && s.cpu_finish_s(v)<=nowS+1e-10
        s.tasks(tid).cpu_finish_ms=s.cpu_finish_s(v)*1000;
        s.cpu_active(v)=0; s.cpu_finish_s(v)=inf; cpuBecameFree=true;
        idealCommunication=isfield(cfg.compute,'ideal_communication') && cfg.compute.ideal_communication;
        if s.tasks(tid).action=="local" || idealCommunication
            s.tasks(tid).result_delivered_ms=s.tasks(tid).cpu_finish_ms;
            if s.tasks(tid).result_delivered_ms<=s.tasks(tid).deadline_ms
                s.tasks(tid).status="completed";
                s=recordReward(s,simParams.seed,tid,nowMs,1);
            else
                s.tasks(tid).status="late_cpu";
            end
        else
            [s,items]=makeFragments(s,tid,v,s.tasks(tid).origin,s.tasks(tid).output_bytes,"result",nowMs);
            s.tasks(tid).output_fragments=numel(items); s.pending_tx{v}=[s.pending_tx{v};items];
        end
    end
end

% Release tasks. Legacy policies decide immediately; joint policies enqueue.
newPendingRelease=false;
while s.release_cursor<=numel(s.tasks) && s.tasks(s.release_cursor).release_ms<=nowMs+1e-6
    tid=s.release_cursor; origin=s.tasks(tid).origin;
    if s.use_joint_scheduler
        s.tasks(tid).status="pending_unassigned";
        s.pending_task_ids(end+1)=tid; %#ok<AGROW>
        s.release_cursor=s.release_cursor+1;
        newPendingRelease=true;
        continue
    end
    candidates=helperCandidates(origin,ids,positionManagement,cfg.compute.helper_range_m,cfg.compute.max_helpers);
    [action,helper,risk,cbr,diagnostic,s]=chooseOverlayAction(s,origin,candidates,stationManagement,positionManagement);
    if action=="offload", s.offload_count=s.offload_count+1; end
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
        cfg.compute.deadline_ms,char(s.policy),char(s.world),diagnostic.local_finish_ms, ...
        diagnostic.best_remote_finish_ms,diagnostic.local_slack_ms, ...
        diagnostic.best_remote_slack_ms,char(diagnostic.feasibility_class)};
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

% P5 joint schedule epochs: coalesce releases into schedule_interval_ms steps.
if s.use_joint_scheduler && ~isempty(s.pending_task_ids) && ...
        (nowMs+1e-9>=s.next_schedule_ms || cpuBecameFree || newPendingRelease)
    [s,epochLog]=spv2xScheduleEpoch(s,s.pending_task_ids,stationManagement,positionManagement,nowMs,s.policy);
    s=logEpoch(s,simParams.seed,epochLog);
    madeProgress=false; deferSeen=false;
    for ai=1:numel(epochLog.actions)
        act=epochLog.actions{ai};
        if ~isfield(act,'type'), continue; end
        at=lower(strtrim(string(act.type)));
        if ~isscalar(at) || strlength(at)==0, at="stop"; else, at=at(1); end
        if at=="assign" || at=="reject"
            madeProgress=true;
        end
        if at=="defer"
            deferSeen=true;
        end
        % Trajectory exposure accounting for P8.C (offload admissions only).
        if at=="assign" && act.task_id>0 && act.task_id<=numel(s.tasks) && s.tasks(act.task_id).action=="offload"
            if ~isfield(s,'traj_exposure_spent'), s.traj_exposure_spent=0; end
            step=0.002;
            if isfield(act,'exposure_marginal') && isfinite(act.exposure_marginal)
                step=max(step,double(act.exposure_marginal));
            end
            s.traj_exposure_spent=min(1,s.traj_exposure_spent+step);
        end
    end
    % Remaining episode fraction for trajectory features.
    if isfield(s,'cfg') && isfield(s.cfg,'compute') && isfield(s.cfg.compute,'task_stop_ms')
        t0=0; if isfield(s.cfg.compute,'task_start_ms'), t0=double(s.cfg.compute.task_start_ms); end
        t1=double(s.cfg.compute.task_stop_ms); den=max(1,t1-t0);
        s.traj_time_frac_remaining=max(0,min(1,(t1-nowMs)/den));
    end
    % Pending set retains only tasks still unassigned after STOP/DEFER.
    stillPending=[];
    for pi=1:numel(s.pending_task_ids)
        tid=s.pending_task_ids(pi);
        if s.tasks(tid).status~="pending_unassigned", continue; end
        % Expire overdue unassigned tasks so they cannot accumulate forever.
        if nowMs+1e-9>=s.tasks(tid).deadline_ms
            s.tasks(tid).action="reject"; s.tasks(tid).status="rejected";
            madeProgress=true;
            continue
        end
        stillPending(end+1)=tid; %#ok<AGROW>
    end
    s.pending_task_ids=stillPending;
    if isempty(stillPending)
        s.next_schedule_ms=nowMs+s.schedule_interval_ms;
    elseif madeProgress
        s.next_schedule_ms=nowMs+s.schedule_interval_ms;
    elseif deferSeen
        % Finite wake so DEFER can retry after radio commitments drain.
        deferMs=40;
        if isfield(s,'p8_traj') && isfield(s.p8_traj,'defer_interval_ms')
            deferMs=double(s.p8_traj.defer_interval_ms);
        end
        s.next_schedule_ms=nowMs+deferMs;
    else
        % STOP-only with live leftovers: wait for CPU free or a new release.
        % Polling every 5 ms rebuilt full observations and hung dens50 WiLab runs.
        s.next_schedule_ms=inf;
    end
end

% Start idle non-preemptive EDF processors.
for v=ids'
    if s.cpu_active(v)==0 && ~isempty(s.cpu_queue{v})
        q=s.cpu_queue{v};
        if s.policy=="fcfs_local"
            ix=1;
        else
            deadlines=arrayfun(@(id)s.tasks(id).deadline_ms,q); [~,ix]=min(deadlines);
        end
        tid=q(ix); q(ix)=[]; s.cpu_queue{v}=q;
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
nextSched=inf;
% Only arm a schedule wake when there is pending work and a finite due time.
% Using next_schedule_ms=0 while awaiting the first release made next_tick_s=0
% and busy-looped WiLab without advancing time (P6 eval hung for hours).
if s.use_joint_scheduler && ~isempty(s.pending_task_ids) && isfinite(s.next_schedule_ms)
    nextSched=s.next_schedule_ms/1000;
end
s.next_tick_s=min([nextRelease,nextCpu,nextSched]);
stationManagement.spv2x=s;
end

function s=recordReward(s,seed,taskId,timeMs,value)
if ~isfield(s,'reward_rows'), s.reward_rows=cell(0,5); end
s.reward_rows(end+1,:)={seed,taskId,timeMs,value,"completion"}; %#ok<AGROW>
end

function s=logEpoch(s,seed,epochLog)
if ~isfield(s,'epoch_rows'), s.epoch_rows=cell(0,8); end
if ~isfield(s,'assignment_feature_rows'), s.assignment_feature_rows=cell(0,20); end
for ai=1:numel(epochLog.actions)
    act=epochLog.actions{ai}; obs=epochLog.observations{ai};
    s.decision_counter=s.decision_counter+1;
    s.epoch_rows(end+1,:)={seed,s.decision_counter,epochLog.epoch_ms,char(act.type), ...
        act.task_id,act.processor_id,numel(obs.tasks),numel(obs.assignments)}; %#ok<AGROW>
    if act.task_id>0, s.tasks(act.task_id).decision_id=s.decision_counter; end
    at=lower(strtrim(string(act.type)));
    if ~isscalar(at) || strlength(at)==0, at="stop"; else, at=at(1); end
    % Skip full candidate dumps on STOP; they dominated wall time when pending grew.
    if at=="assign" || at=="reject"
    for k=1:numel(obs.assignments)
        a=obs.assignments(k);
        chosen=at=="assign" && a.task_id==act.task_id && a.processor_id==act.processor_id;
        tIdx=find([obs.tasks.task_id]==a.task_id,1); cls="NA";
        if ~isempty(tIdx), cls=char(obs.tasks(tIdx).feasibility_class); end
        s.assignment_feature_rows(end+1,:)={seed,s.decision_counter,epochLog.epoch_ms,a.task_id,a.processor_id, ...
            a.is_local,a.processor_cpu_ghz,a.residual_workload_mcycles,a.estimated_input_latency_ms, ...
            a.estimated_result_latency_ms,a.estimated_helper_queueing_ms,a.estimated_computation_latency_ms, ...
            a.predicted_remote_deadline_slack_ms,a.predicted_completion_probability, ...
            a.predicted_conditional_safety_risk,a.relative_distance_m,a.estimated_resource_block_slots, ...
            a.masked,char(a.mask_reason),cls,chosen}; %#ok<AGROW>
    end
    end
    % Legacy decisions.csv compatibility for chosen ASSIGN/REJECT.
    if act.type=="ASSIGN" || act.type=="REJECT"
        tid=act.task_id; origin=s.tasks(tid).origin;
        tIdx=find([obs.tasks.task_id]==tid,1);
        localFinish=NaN; remoteFinish=NaN; localSlack=NaN; remoteSlack=NaN; cls="NA";
        if ~isempty(tIdx)
            localSlack=obs.tasks(tIdx).local_deadline_slack_ms;
            remoteSlack=obs.tasks(tIdx).best_remote_slack_ms;
            remoteFinish=obs.tasks(tIdx).best_remote_finish_ms;
            localFinish=s.tasks(tid).deadline_ms-localSlack;
            cls=char(obs.tasks(tIdx).feasibility_class);
        end
        actionStr="reject"; helper=0;
        if act.type=="ASSIGN"
            if act.processor_id==origin, actionStr="local"; else, actionStr="offload"; helper=act.processor_id; end
        end
        s.decision_rows(end+1,:)={seed,s.decision_counter,epochLog.epoch_ms,origin,tid,actionStr,helper,0, ...
            0,obs.comm.recent_cbr,0,NaN,0,0,s.cfg.compute.input_bytes,s.cfg.compute.output_bytes, ...
            s.cfg.compute.cycles_per_task,s.cfg.compute.deadline_ms,char(s.policy),char(s.world), ...
            localFinish,remoteFinish,localSlack,remoteSlack,cls}; %#ok<AGROW>
    end
end
end

function candidates=helperCandidates(origin,ids,positionManagement,maxRange,maxHelpers)
if ~isfield(positionManagement,'distanceReal'), candidates=[]; return; end
d=positionManagement.distanceReal(origin,ids); valid=ids(d>0 & d<=maxRange);
if isempty(valid), candidates=[]; return; end
[~,order]=sort(positionManagement.distanceReal(origin,valid)); valid=valid(order);
candidates=valid(1:min(maxHelpers,numel(valid)))';
end

function [action,helper,risk,cbr,diagnostic,s]=chooseOverlayAction(s,origin,candidates,stationManagement,positionManagement)
cfg=s.cfg; helper=0; cbr=0;
if isfield(stationManagement,'cbrCV2Xvalues')
    values=stationManagement.cbrCV2Xvalues(origin,:); values=values(values>=0);
    if ~isempty(values), cbr=values(end); end
end
risk=min(1,0.002+0.015*cbr+0.003*(cfg.compute.input_bytes/cfg.radio.max_transport_block_bytes));
nowS=s.tasks(s.release_cursor).release_ms/1000;
deadlineMs=s.tasks(s.release_cursor).deadline_ms;
localFinishMs=forecastCpuFinishMs(s,origin,nowS,cfg.compute.cycles_per_task);
helperFinishMs=inf(size(candidates));
helperFinishDeployable=inf(size(candidates));
for ci=1:numel(candidates)
    helperFinishMs(ci)=forecastCpuFinishMs(s,candidates(ci),nowS,cfg.compute.cycles_per_task);
    % Deployable: no helper-queue / active-residual leakage (P5 strip).
    helperFinishDeployable(ci)=forecastCpuFinishMsDeployable(s,candidates(ci),nowS,cfg.compute.cycles_per_task);
end
if isempty(helperFinishMs)
    bestHelperFinish=inf; bestIndex=0;
    bestHelperFinishDep=inf; bestIndexDep=0;
else
    [bestHelperFinish,bestIndex]=min(helperFinishMs);
    [bestHelperFinishDep,bestIndexDep]=min(helperFinishDeployable);
end
fragments=ceil(cfg.compute.input_bytes/cfg.radio.max_transport_block_bytes)+ ...
    ceil(cfg.compute.output_bytes/cfg.radio.max_transport_block_bytes);
remoteFinishMs=bestHelperFinish+fragments*cfg.radio.selection_window_ms;
remoteFinishDeployableMs=bestHelperFinishDep+fragments*cfg.radio.selection_window_ms;
localSlack=deadlineMs-localFinishMs; remoteSlack=deadlineMs-remoteFinishMs;
if localSlack>=0 && remoteSlack<0, feasibility="I";
elseif localSlack>=0 && remoteSlack>=0, feasibility="II";
elseif localSlack<0 && remoteSlack>=0, feasibility="III";
else, feasibility="IV";
end
diagnostic=struct('local_finish_ms',localFinishMs,'best_remote_finish_ms',remoteFinishMs, ...
    'local_slack_ms',localSlack,'best_remote_slack_ms',remoteSlack, ...
    'feasibility_class',feasibility);
if s.world=="baseline" || ismember(s.policy,["no_offload","fcfs_local","deadline_local"]) || isempty(candidates)
    action="local"; return
end
switch s.policy
    case {"oracle_instant","oracle_actual","safety_oracle"}
        policyRemoteFinish=remoteFinishMs;
        if s.policy=="oracle_instant", policyRemoteFinish=bestHelperFinish; end
        safetyFeasible=s.policy~="safety_oracle" || ...
            risk<=cfg.control.risk_harm_limit;
        if s.policy=="safety_oracle" && isfield(cfg.compute,'oracle_max_offloads')
            admitted=sum(arrayfun(@(t)t.action=="offload",s.tasks));
            safetyFeasible=safetyFeasible && admitted<cfg.compute.oracle_max_offloads;
        end
        if localFinishMs<=deadlineMs
            action="local";
        elseif policyRemoteFinish<=deadlineMs && safetyFeasible
            action="offload"; helper=candidates(bestIndex);
        else
            action="local";
        end
    case "oracle_deployable"
        % P9 diagnostic: identical local-first rescue rule as oracle_actual /
        % radio_search, but helper finish uses deployable (queue-blind) forecasts.
        if localFinishMs<=deadlineMs
            action="local";
        elseif remoteFinishDeployableMs<=deadlineMs && bestIndexDep>0
            action="offload"; helper=candidates(bestIndexDep);
        else
            action="local";
        end
    case "local_first_rescue"
        if localFinishMs<=deadlineMs
            action="local";
        elseif remoteFinishMs<=deadlineMs
            action="offload"; helper=candidates(bestIndex);
        else
            action="reject";
        end
    case "min_completion_time"
        if localFinishMs<=min(remoteFinishMs,deadlineMs)
            action="local";
        elseif remoteFinishMs<=deadlineMs
            action="offload"; helper=candidates(bestIndex);
        else
            action="reject";
        end
    case "safety_budget"
        % Approximate privileged safety search candidate: local-first task
        % rescue whose cumulative admissions are capped at K offloads and
        % released evenly over the task horizon (token bucket), so a budget
        % does not simply consume its tokens in the first seconds.
        assert(isfield(cfg.compute,'safety_budget_offloads'),'Tick:Budget','safety_budget_offloads missing');
        budget=cfg.compute.safety_budget_offloads;
        if localFinishMs<=deadlineMs
            action="local";
        elseif remoteFinishMs<=deadlineMs
            [startMs,stopMs]=taskHorizonMs(cfg);
            progress=min(1,max(0,(s.tasks(s.release_cursor).release_ms-startMs)/(stopMs-startMs)));
            allowed=min(budget,floor(budget*progress)+1);
            if s.offload_count<allowed
                action="offload"; helper=candidates(bestIndex);
            else
                action="local";
            end
        else
            action="local";
        end
    case "risk_myopic"
        % Provisional conservative admission.  Local execution is kept when
        % it finishes no later than remote execution.  A remote admission
        % must (i) be deadline feasible and (ii) fit inside a pooled
        % predicted-harm budget, charged with a risk that is conditional on
        % the offloads already committed in the action window.
        [riskCond,cost,fits]=conditionalRiskGovernor(s,risk);
        risk=riskCond;
        if localFinishMs<=min(remoteFinishMs,deadlineMs)
            action="local";
        elseif remoteFinishMs<=deadlineMs && fits
            action="offload"; helper=candidates(bestIndex);
            s.governor_spent=s.governor_spent+cost;
        elseif localFinishMs<=deadlineMs
            action="local";
        else
            action="reject";
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

function [startMs,stopMs]=taskHorizonMs(cfg)
assert(isfield(cfg.compute,'task_start_ms') && isfield(cfg.compute,'task_stop_ms'),'Tick:Horizon', ...
    'task_start_ms/task_stop_ms are required by budgeted policies');
startMs=double(cfg.compute.task_start_ms); stopMs=double(cfg.compute.task_stop_ms);
end

function [riskCond,cost,fits]=conditionalRiskGovernor(s,baseRisk)
%CONDITIONALRISKGOVERNOR Union-bound risk given committed offloads.
% r(a|B) = r0 (1+|B|): every offload still inside its action window is
% assumed to interact with the proposed one, so jointly admitted offloads are
% never treated as independent.  The admission is charged r(a|B)*W against a
% pooled budget  margin * delta * T  (delta is the policy-level harm limit,
% T the task horizon).  margin is fixed a priori and is not tuned.
cfg=s.cfg; windowMs=cfg.compute.deadline_ms; nowMs=s.tasks(s.release_cursor).release_ms;
committed=0; k=s.release_cursor-1;
while k>=1 && s.tasks(k).release_ms>nowMs-windowMs
    if s.tasks(k).action=="offload", committed=committed+1; end
    k=k-1;
end
riskCond=min(1,baseRisk*(1+committed));
cost=riskCond*windowMs/1000;
[startMs,stopMs]=taskHorizonMs(cfg);
margin=0.5;
if isfield(cfg.control,'risk_governor_margin'), margin=cfg.control.risk_governor_margin; end
budget=margin*cfg.control.risk_harm_limit*(stopMs-startMs)/1000;
fits=s.governor_spent+cost<=budget;
end

function finishMs=forecastCpuFinishMs(s,vehicle,nowS,newCycles)
backlogCycles=sum(arrayfun(@(id)s.tasks(id).cycles,s.cpu_queue{vehicle}));
if s.cpu_active(vehicle)>0
    backlogCycles=backlogCycles+max(0,s.cpu_finish_s(vehicle)-nowS)*s.cpu_rate(vehicle);
end
finishMs=nowS*1000+(backlogCycles+newCycles)/s.cpu_rate(vehicle)*1000;
end

function finishMs=forecastCpuFinishMsDeployable(s,vehicle,nowS,newCycles)
% No helper-queue or active-residual information (matches P5 deployable strip).
finishMs=nowS*1000+newCycles/s.cpu_rate(vehicle)*1000;
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

