function plan=spv2xLearningAssistedMyopicPlan(obs,s,tentative)
%SPV2XLEARNINGASSISTEDMYOPICPLAN Frozen risk-constrained myopic + learned V̂(O').
% Maximizes sum p̂ + λ * V̂_θ(post-decision context) under the same masks and
% cumulative risk shield as the calibrated myopic. Without a value model,
% identical to spv2xRiskConstrainedMyopicPlan.
if nargin<3 || isempty(tentative), tentative=struct('offload_count',0,'extra_cycles',[]); end
harmLimit=0.01;
if isfield(s,'cfg') && isfield(s.cfg,'control') && isfield(s.cfg.control,'risk_harm_limit')
    harmLimit=double(s.cfg.control.risk_harm_limit);
end
lambda=0.25;
if isfield(s,'cfg') && isfield(s.cfg,'control') && isfield(s.cfg.control,'value_assist_lambda')
    lambda=double(s.cfg.control.value_assist_lambda);
end
hasV=isfield(s,'p8_value') && isstruct(s.p8_value) && isfield(s.p8_value,'type') && ...
    strlength(string(s.p8_value.type))>0 && string(s.p8_value.type)=="value_assist";
if ~hasV
    plan=spv2xRiskConstrainedMyopicPlan(obs,s,tentative);
    return
end
plan=struct('type',"STOP",'task_id',0,'processor_id',0); plan=plan(1:0);
if isempty(obs.tasks), return; end
pending=unique([obs.tasks.task_id],'stable');
candidates=obs.assignments;
if isfield(candidates,'masked'), candidates=candidates(~[candidates.masked]); end
if isempty(candidates), return; end
if numel(pending)>3
    plan=greedyValue(obs,pending,candidates,harmLimit,tentative,s.p8_value,lambda);
else
    plan=exhaustiveValue(obs,pending,candidates,harmLimit,tentative,s.p8_value,lambda);
end
plan=sanitizeAssistPlan(plan);
end

function plan=sanitizeAssistPlan(plan)
if isempty(plan), return; end
keep=false(numel(plan),1);
for i=1:numel(plan)
    if ~isfield(plan(i),'type') || ~isfield(plan(i),'task_id'), continue; end
    t=string(plan(i).type);
    keep(i)=isscalar(t) && strlength(t)>0 && plan(i).task_id>0 && ...
        ismember(t,["ASSIGN","REJECT"]);
end
plan=plan(keep);
end

function plan=exhaustiveValue(obs,pending,candidates,harmLimit,tentative,vmodel,lambda)
choices=cell(numel(pending),1);
for i=1:numel(pending)
    opts=candidates([candidates.task_id]==pending(i));
    keep=opts([opts.predicted_completion_probability]>0 | [opts.is_local]);
    if isempty(keep), keep=opts; end
    choices{i}=keep;
end
bestScore=-inf;
bestPlan=struct('type',"ASSIGN",'task_id',0,'processor_id',0); bestPlan=bestPlan(1:0);
idx=ones(1,numel(pending)); done=false;
while ~done
    [score,candPlan,feasible]=scoreChoiceValue(obs,pending,choices,idx,harmLimit,tentative,vmodel,lambda);
    if feasible && (score>bestScore || (score==bestScore && numel(candPlan)>numel(bestPlan)))
        bestScore=score; bestPlan=candPlan;
    end
    k=numel(pending);
    while k>=1
        nOpts=numel(choices{k})+1;
        idx(k)=idx(k)+1;
        if idx(k)<=max(1,nOpts), break; end
        idx(k)=1; k=k-1;
    end
    if k<1, done=true; end
end
plan=bestPlan;
end

function [score,plan,feasible]=scoreChoiceValue(obs,pending,choices,idx,harmLimit,tentative,vmodel,lambda)
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
    if isfield(a,'predicted_completion_probability') && isfinite(a.predicted_completion_probability)
        pHat=max(pHat,double(a.predicted_completion_probability));
        pHat=min(1,pHat);
    end
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
    else, extra(a.processor_id)=cycles; end
    plan(end+1,1)=struct('type',"ASSIGN",'task_id',tid,'processor_id',a.processor_id); %#ok<AGROW>
end
% Post-decision value: remaining unfinished after this schedule ≈ 0 in exhaustive full cover.
% Use context of committed load as V̂ input.
nLeft=0;
for ii=1:numel(plan)
    if string(plan(ii).type)=="REJECT", nLeft=nLeft+1; end
end
ctx=[nLeft,committed,score,numel(plan),0,0];
score=score+lambda*forwardV(vmodel,ctx);
end

function plan=greedyValue(obs,pending,candidates,harmLimit,tentative,vmodel,lambda)
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
        nextCommitted=committed+double(~a.is_local);
        ctx=[numel(remaining)-1,nextCommitted,pHat,1,a.predicted_conditional_safety_risk,a.predicted_completion_probability];
        sc=pHat+lambda*forwardV(vmodel,ctx);
        if sc>bestScore, bestScore=sc; best=a; end
    end
    if isempty(best) || bestScore<=-1e8
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
    else, extra(best.processor_id)=cycles; end
    candidates=candidates([candidates.task_id]~=best.task_id);
end
end

function v=forwardV(model,x)
means=double(model.means(:)'); scales=double(model.scales(:)');
scales(scales<1e-6)=1;
z=x(:)';
if numel(z)<numel(means), z(end+1:numel(means))=0; end
z=z(1:numel(means));
z=(z'-means)./scales; z(~isfinite(z))=0;
h=max(0,double(model.hidden_weights)*z'+double(model.hidden_bias(:)));
v=double(model.output_weights)*h+double(model.output_bias(:));
v=v(1);
end
