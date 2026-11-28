function plan=spv2xRiskConstrainedMyopicPlan(obs,s,tentative)
%SPV2XRISKCONSTRAINEDMYOPICPLAN Risk-constrained approximate myopic schedule.
% Solves max sum_ij x_ij p_hat_ij s.t. unique tasks, masks, conditional risk
% <= delta, using deployable observations only (P6 information fairness).
% Exact enumeration is used for |K|<=3 pending tasks; larger epochs use a
% greedy commitment-aware heuristic.  Treat results as approximate myopic
% performance, not a claim of global optimality for large pending sets.
if nargin<3 || isempty(tentative), tentative=struct('offload_count',0,'extra_cycles',[]); end
harmLimit=0.01;
if isfield(s,'cfg') && isfield(s.cfg,'control') && isfield(s.cfg.control,'risk_harm_limit')
    harmLimit=double(s.cfg.control.risk_harm_limit);
end
plan=struct('type',"STOP",'task_id',0,'processor_id',0); plan=plan(1:0);
if isempty(obs.tasks), return; end
pending=unique([obs.tasks.task_id],'stable');
candidates=obs.assignments;
if isfield(candidates,'masked')
    candidates=candidates(~[candidates.masked]);
end
if isempty(candidates), return; end
if numel(pending)<=3
    plan=exhaustivePlan(obs,pending,candidates,harmLimit,tentative);
else
    plan=greedyPlan(obs,pending,candidates,harmLimit,tentative);
end
plan=sanitizePlan(plan);
end

function plan=sanitizePlan(plan)
if isempty(plan), return; end
keep=false(numel(plan),1);
for i=1:numel(plan)
    if ~isfield(plan(i),'type') || ~isfield(plan(i),'task_id'), continue; end
    t=string(plan(i).type);
    keep(i)=isscalar(t) && strlength(t)>0 && plan(i).task_id>0;
end
plan=plan(keep);
end

function plan=exhaustivePlan(obs,pending,candidates,harmLimit,tentative)
choices=cell(numel(pending),1);
for i=1:numel(pending)
    opts=candidates([candidates.task_id]==pending(i));
    keep=opts([opts.predicted_completion_probability]>0 | [opts.is_local]);
    if isempty(keep), keep=opts; end
    choices{i}=keep;
end
bestScore=-inf;
bestPlan=struct('type',"ASSIGN",'task_id',0,'processor_id',0); bestPlan=bestPlan(1:0);
idx=ones(1,numel(pending));
done=false;
while ~done
    [score,candPlan,feasible]=scoreChoice(obs,pending,choices,idx,harmLimit,tentative);
    % Prefer any positive-completion schedule over empty/reject-only.
    if feasible && (score>bestScore || (score==bestScore && numel(candPlan)>numel(bestPlan)))
        bestScore=score; bestPlan=candPlan;
    end
    k=numel(pending);
    while k>=1
        nOpts=numel(choices{k})+1; % last index = REJECT
        idx(k)=idx(k)+1;
        if idx(k)<=max(1,nOpts), break; end
        idx(k)=1; k=k-1;
    end
    if k<1, done=true; end
end
plan=bestPlan;
end

function [score,plan,feasible]=scoreChoice(obs,pending,choices,idx,harmLimit,tentative)
score=0; feasible=true;
plan=struct('type',"ASSIGN",'task_id',0,'processor_id',0); plan=plan(1:0);
extra=containers.Map('KeyType','double','ValueType','double');
committed=0;
if isfield(tentative,'offload_count'), committed=tentative.offload_count; end
if isfield(tentative,'extra_cycles') && ~isempty(tentative.extra_cycles)
    for v=1:numel(tentative.extra_cycles)
        if tentative.extra_cycles(v)>0, extra(v)=tentative.extra_cycles(v); end
    end
end
for i=1:numel(pending)
    tid=pending(i); nKeep=numel(choices{i});
    if nKeep==0 || idx(i)==nKeep+1
        plan(end+1,1)=struct('type',"REJECT",'task_id',tid,'processor_id',0); %#ok<AGROW>
        continue
    end
    a=choices{i}(idx(i));
    tIdx=find([obs.tasks.task_id]==tid,1);
    addMs=a.estimated_computation_latency_ms+a.estimated_input_latency_ms+a.estimated_result_latency_ms;
    if ~a.is_local, addMs=addMs+max(0,a.estimated_helper_queueing_ms); end
    queuedMs=0;
    if extra.isKey(a.processor_id) && a.processor_cpu_ghz>0
        queuedMs=extra(a.processor_id)/(a.processor_cpu_ghz*1e9)*1000;
    end
    pHat=double(obs.tasks(tIdx).remaining_deadline_ms-(queuedMs+addMs)>=0);
    if ~a.is_local
        base=a.predicted_conditional_safety_risk;
        if ~isfinite(base) || base<=0, base=0.002; end
        risk=min(1,base*(1+committed));
        if risk>harmLimit+1e-12, feasible=false; score=-inf; return; end
        committed=committed+1;
    end
    score=score+pHat;
    cycles=obs.tasks(tIdx).cycles_mcycles*1e6;
    if extra.isKey(a.processor_id), extra(a.processor_id)=extra(a.processor_id)+cycles;
    else, extra(a.processor_id)=cycles;
    end
    plan(end+1,1)=struct('type',"ASSIGN",'task_id',tid,'processor_id',a.processor_id); %#ok<AGROW>
end
end

function plan=greedyPlan(obs,pending,candidates,harmLimit,tentative)
plan=struct('type',"ASSIGN",'task_id',0,'processor_id',0); plan=plan(1:0);
remaining=pending(:);
committed=0;
if isfield(tentative,'offload_count'), committed=tentative.offload_count; end
extra=containers.Map('KeyType','double','ValueType','double');
while ~isempty(remaining)
    bestScore=-inf; best=[];
    for i=1:numel(candidates)
        a=candidates(i);
        if ~ismember(a.task_id,remaining), continue; end
        tIdx=find([obs.tasks.task_id]==a.task_id,1);
        addMs=a.estimated_computation_latency_ms+a.estimated_input_latency_ms+a.estimated_result_latency_ms;
        if ~a.is_local, addMs=addMs+max(0,a.estimated_helper_queueing_ms); end
        queuedMs=0;
        if extra.isKey(a.processor_id) && a.processor_cpu_ghz>0
            queuedMs=extra(a.processor_id)/(a.processor_cpu_ghz*1e9)*1000;
        end
        pHat=double(obs.tasks(tIdx).remaining_deadline_ms-(queuedMs+addMs)>=0);
        if ~a.is_local
            base=a.predicted_conditional_safety_risk;
            if ~isfinite(base) || base<=0, base=0.002; end
            risk=min(1,base*(1+committed));
            if risk>harmLimit+1e-12, continue; end
        end
        if pHat>bestScore
            bestScore=pHat; best=a;
        end
    end
    if isempty(best) || bestScore<=0
        % Do not leave deadline-expired tasks in pending_unassigned: that caused a
        % 5 ms reschedule storm in full WiLab runs (P6 eval hung for hours).
        for r=1:numel(remaining)
            tid=remaining(r); tIdx=find([obs.tasks.task_id]==tid,1);
            if ~isempty(tIdx) && obs.tasks(tIdx).remaining_deadline_ms<=0
                plan(end+1,1)=struct('type',"REJECT",'task_id',tid,'processor_id',0); %#ok<AGROW>
            end
        end
        break
    end
    plan(end+1,1)=struct('type',"ASSIGN",'task_id',best.task_id,'processor_id',best.processor_id); %#ok<AGROW>
    remaining(remaining==best.task_id)=[];
    if ~best.is_local, committed=committed+1; end
    tIdx=find([obs.tasks.task_id]==best.task_id,1);
    cycles=obs.tasks(tIdx).cycles_mcycles*1e6;
    if extra.isKey(best.processor_id), extra(best.processor_id)=extra(best.processor_id)+cycles;
    else, extra(best.processor_id)=cycles;
    end
    candidates=candidates([candidates.task_id]~=best.task_id);
end
end
