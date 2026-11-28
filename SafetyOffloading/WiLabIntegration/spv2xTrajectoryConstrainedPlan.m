function plan=spv2xTrajectoryConstrainedPlan(obs,s,tentative)
%SPV2XTRAJECTORYCONSTRAINEDPLAN P8.C — trajectory-aware constrained admission.
% One pending task is the common case. Actions: ASSIGN local/offload, REJECT, DEFER.
% Cumulative exposure gate uses models/p8_trajectory_constraint.json when present.
% Does not claim that the gate guarantees P2 safety; matched eval remains authoritative.
if nargin<3 || isempty(tentative), tentative=struct('offload_count',0); end
actions=struct('type',{},'task_id',{},'processor_id',{},'exposure_marginal',{});
if isempty(obs.tasks), plan=actions; return; end

harmLimit=0.01;
if isfield(s,'cfg') && isfield(s.cfg,'control') && isfield(s.cfg.control,'risk_harm_limit')
    harmLimit=double(s.cfg.control.risk_harm_limit);
end
model=[];
if isfield(s,'p8_traj') && isstruct(s.p8_traj) && isfield(s.p8_traj,'type') && ...
        string(s.p8_traj.type)=="trajectory_constraint"
    model=s.p8_traj;
end

% Trajectory snapshot (deployable / simulator-visible).
traj=trajectorySnapshot(obs,s,tentative);
spent=0;
if isfield(s,'traj_exposure_spent'), spent=double(s.traj_exposure_spent); end

% Process each pending task once (usually one).
pending=unique([obs.tasks.task_id],'stable');
assigned=containers.Map('KeyType','double','ValueType','logical');
for ti=1:numel(pending)
    tid=pending(ti);
    if assigned.isKey(tid), continue; end
    tIdx=find([obs.tasks.task_id]==tid,1);
    cands=obs.assignments([obs.assignments.task_id]==tid);
    if isfield(cands,'masked'), cands=cands(~[cands.masked]); end
    if isempty(cands)
        if obs.tasks(tIdx).remaining_deadline_ms<=0
            actions(end+1,1)=struct('type',"REJECT",'task_id',tid,'processor_id',0,'exposure_marginal',0); %#ok<AGROW>
        else
            actions(end+1,1)=struct('type',"DEFER",'task_id',tid,'processor_id',0,'exposure_marginal',0); %#ok<AGROW>
        end
        continue
    end

    local=[]; remote=cands([cands.is_local]==0);
    locals=cands([cands.is_local]==1);
    if ~isempty(locals), local=locals(1); end

    localOk=~isempty(local) && local.predicted_completion_probability>0;
    % Best remote by completion then risk.
    bestR=[]; bestScore=-inf;
    for i=1:numel(remote)
        a=remote(i);
        marg=marginalExposure(model,traj,a,s);
        if spent+marg>harmLimit+1e-12, continue; end
        sc=double(a.predicted_completion_probability)-0.05*double(a.predicted_conditional_safety_risk);
        if sc>bestScore, bestScore=sc; bestR=a; end
    end
    remoteOk=~isempty(bestR) && bestR.predicted_completion_probability>0;
    localP=0; if localOk, localP=double(local.predicted_completion_probability); end

    if localOk && (~remoteOk || localP+1e-9>=bestR.predicted_completion_probability)
        actions(end+1,1)=struct('type',"ASSIGN",'task_id',tid,'processor_id',local.processor_id,'exposure_marginal',0); %#ok<AGROW>
        assigned(tid)=true;
        continue
    end
    if remoteOk
        marg=marginalExposure(model,traj,bestR,s);
        actions(end+1,1)=struct('type',"ASSIGN",'task_id',tid,'processor_id',bestR.processor_id,'exposure_marginal',marg); %#ok<AGROW>
        assigned(tid)=true;
        spent=spent+marg;
        continue
    end
    % Offload blocked by cumulative exposure or infeasible.
    if localOk
        actions(end+1,1)=struct('type',"ASSIGN",'task_id',tid,'processor_id',local.processor_id,'exposure_marginal',0); %#ok<AGROW>
        assigned(tid)=true;
        continue
    end
    % Cannot complete now: DEFER if radio/commitment hot and time remains; else REJECT.
    if shouldDefer(model,traj,obs.tasks(tIdx))
        actions(end+1,1)=struct('type',"DEFER",'task_id',tid,'processor_id',0,'exposure_marginal',0); %#ok<AGROW>
    else
        actions(end+1,1)=struct('type',"REJECT",'task_id',tid,'processor_id',0,'exposure_marginal',0); %#ok<AGROW>
        assigned(tid)=true;
    end
end
plan=actions;
end

function traj=trajectorySnapshot(obs,s,tentative)
traj=struct('recent_cbr',0,'outstanding_tx',0,'cum_offloads',0,'remaining_frac',1, ...
    'mean_selection_delay_proxy',0);
if isfield(obs,'comm')
    if isfield(obs.comm,'recent_cbr'), traj.recent_cbr=double(obs.comm.recent_cbr); end
    oi=0; oj=0;
    if isfield(obs.comm,'outstanding_input_blocks'), oi=double(obs.comm.outstanding_input_blocks); end
    if isfield(obs.comm,'outstanding_result_blocks'), oj=double(obs.comm.outstanding_result_blocks); end
    traj.outstanding_tx=oi+oj;
end
if isfield(s,'offload_count'), traj.cum_offloads=double(s.offload_count); end
if isfield(tentative,'offload_count'), traj.cum_offloads=traj.cum_offloads+double(tentative.offload_count); end
if isfield(s,'traj_time_frac_remaining'), traj.remaining_frac=double(s.traj_time_frac_remaining); end
if isfield(obs,'assignments') && ~isempty(obs.assignments)
    a=obs.assignments;
    if isfield(a,'predicted_remote_deadline_slack_ms')
        sl=[a.predicted_remote_deadline_slack_ms];
        traj.mean_selection_delay_proxy=mean(max(0,-sl))/100;
    end
end
end

function marg=marginalExposure(model,traj,a,s)
base=double(a.predicted_conditional_safety_risk);
if ~isfinite(base) || base<=0, base=0.002; end
if isempty(model)
    % Fallback: trajectory-inflated action risk (still cumulative via caller spent).
    loadFactor=1+0.5*traj.recent_cbr+0.05*traj.outstanding_tx+0.02*traj.cum_offloads;
    marg=min(1,base*loadFactor);
    return
end
x=[traj.recent_cbr, traj.outstanding_tx, traj.cum_offloads/50.0, base, ...
    traj.remaining_frac, traj.mean_selection_delay_proxy];
means=double(model.means(:)'); scales=double(model.scales(:)');
scales(scales<1e-6)=1;
if numel(x)<numel(means), x(end+1:numel(means))=0; end
x=x(1:numel(means));
z=(x-means)./scales; z(~isfinite(z))=0;
w=double(model.weights(:));
p=1/(1+exp(-(w(1)+sum(w(2:end).*z(:)))));
scale=1.0;
if isfield(model,'marginal_scale'), scale=double(model.marginal_scale); end
marg=min(1,max(base,p*scale));
% Inflate further when radio commitments are already high.
marg=min(1,marg*(1+0.25*traj.recent_cbr+0.03*min(traj.outstanding_tx,20)));
end

function tf=shouldDefer(model,traj,task)
tf=false;
if task.remaining_deadline_ms<=10, return; end
cbrTh=0.25; outTh=8;
if ~isempty(model)
    if isfield(model,'defer_cbr_threshold'), cbrTh=double(model.defer_cbr_threshold); end
    if isfield(model,'defer_outstanding_threshold'), outTh=double(model.defer_outstanding_threshold); end
end
tf=(traj.recent_cbr>=cbrTh || traj.outstanding_tx>=outTh) && traj.remaining_frac>0.05;
end
