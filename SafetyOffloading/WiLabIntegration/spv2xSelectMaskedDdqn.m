function action=spv2xSelectMaskedDdqn(obs,s,tentative)
%SPV2XSELECTMASKEDDDQN Masked Double-DQN action over ASSIGN / REJECT / STOP.
action=struct('type',"STOP",'task_id',0,'processor_id',0);
if isempty(obs.tasks), return; end
harmLimit=0.01;
if isfield(s,'cfg') && isfield(s.cfg,'control') && isfield(s.cfg.control,'risk_harm_limit')
    harmLimit=double(s.cfg.control.risk_harm_limit);
end
committed=0;
if isfield(tentative,'offload_count'), committed=double(tentative.offload_count); end

cands=obs.assignments;
if isfield(cands,'masked'), cands=cands(~[cands.masked]); end
keep=true(numel(cands),1);
for i=1:numel(cands)
    if cands(i).is_local, continue; end
    base=cands(i).predicted_conditional_safety_risk;
    if ~isfinite(base) || base<=0, base=0.002; end
    if min(1,base*(1+committed))>harmLimit+1e-12, keep(i)=false; end
end
cands=cands(keep);

model=[];
if isfield(s,'p8_ddqn') && isstruct(s.p8_ddqn) && isfield(s.p8_ddqn,'type') && string(s.p8_ddqn.type)=="masked_ddqn"
    model=s.p8_ddqn;
end
if isempty(model) || ~isfield(model,'means')
    for i=1:numel(cands)
        if cands(i).is_local && cands(i).predicted_completion_probability>0
            action=struct('type',"ASSIGN",'task_id',cands(i).task_id,'processor_id',cands(i).processor_id);
            return
        end
    end
    if obs.tasks(1).remaining_deadline_ms<=0
        action=struct('type',"REJECT",'task_id',obs.tasks(1).task_id,'processor_id',0);
    end
    return
end

ctx=contextFeatures(obs,committed);
bestQ=-inf; bestAction=action;
featNames=string(model.features);
for i=1:numel(cands)
    x=[ctx, assignmentFeatures(cands(i),featNames)];
    q=forwardHeads(model,x);
    if q(1)>bestQ
        bestQ=q(1);
        bestAction=struct('type',"ASSIGN",'task_id',cands(i).task_id,'processor_id',cands(i).processor_id);
    end
end
xSpecial=[ctx, zeros(1,numel(featNames))];
q=forwardHeads(model,xSpecial);
if q(2)>bestQ
    bestQ=q(2);
    bestAction=struct('type',"REJECT",'task_id',obs.tasks(1).task_id,'processor_id',0);
end
if q(3)>bestQ
    bestAction=struct('type',"STOP",'task_id',0,'processor_id',0);
end
action=bestAction;
end

function ctx=contextFeatures(obs,committed)
nPend=numel(obs.tasks);
nCand=0; if isfield(obs,'assignments'), nCand=numel(obs.assignments); end
meanP=0; meanR=0; meanSlack=0;
if isfield(obs,'assignments') && ~isempty(obs.assignments)
    a=obs.assignments; if isfield(a,'masked'), a=a(~[a.masked]); end
    if ~isempty(a)
        meanP=mean([a.predicted_completion_probability]);
        meanR=mean([a.predicted_conditional_safety_risk]);
        if isfield(a,'predicted_remote_deadline_slack_ms')
            meanSlack=mean([a.predicted_remote_deadline_slack_ms]);
        elseif isfield(a,'predicted_deadline_slack_ms')
            meanSlack=mean([a.predicted_deadline_slack_ms]);
        end
    end
end
ctx=[nPend,nCand,committed,meanP,meanR,meanSlack];
end

function x=assignmentFeatures(a,featNames)
x=zeros(1,numel(featNames));
for i=1:numel(featNames)
    f=featNames(i);
    if isfield(a,char(f)), x(i)=double(a.(char(f)));
    elseif f=="predicted_deadline_slack_ms" && isfield(a,'predicted_remote_deadline_slack_ms')
        x(i)=double(a.predicted_remote_deadline_slack_ms);
    end
end
end

function q=forwardHeads(model,x)
means=double(model.means(:)'); scales=double(model.scales(:)');
scales(scales<1e-6)=1;
if numel(x)<numel(means), x(end+1:numel(means))=0; end
x=x(1:numel(means));
z=(x-means)./scales; z(~isfinite(z))=0;
h=max(0,double(model.hidden_weights)*z'+double(model.hidden_bias(:)));
if isfield(model,'full_output_weights')
    W=double(model.full_output_weights); b=double(model.full_output_bias(:));
    q=W*h+b;
else
    q0=double(model.output_weights)*h+double(model.output_bias(:));
    q=[q0(1); q0(1); q0(1)];
    if isfield(model,'head_bias')
        if isfield(model.head_bias,'reject'), q(2)=q(1)+double(model.head_bias.reject); end
        if isfield(model.head_bias,'stop'), q(3)=q(1)+double(model.head_bias.stop); end
    end
end
if numel(q)<3, q(end+1:3)=-inf; end
end
