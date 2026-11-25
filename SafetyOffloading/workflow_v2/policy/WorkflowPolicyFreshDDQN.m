function [action,X]=WorkflowPolicyFreshDDQN(W)
%WORKFLOWPOLICYFRESHDDQN Candidate-index fresh-action policy entrypoint.
c=WorkflowCandidateBuilder(W); X=WorkflowFreshCandidateFeatures(W,c);
score=bcScore(W,X);
if isempty(score)
    % Safe deterministic bootstrap until BC weights are supplied.
    score=2*X(:,8)-0.15*X(:,6)-0.10*X(:,9)+0.05*X(:,10);
end
score(categoricalKind(c,"STOP_PLAN"))=-1e6;
% Rejecting a workflow is dominated whenever its first eligible stage can be
% committed locally.  Keep the learned score from selecting this destructive
% action through unsupported extrapolation in offline TD updates.
for i=1:numel(c)
    if c(i).kind~="REJECT_WORKFLOW", continue; end
    for j=1:numel(c)
        if c(j).kind=="COMMIT_LOCAL" && c(j).workflow_id==c(i).workflow_id
            score(i)=-1e6; break
        end
    end
end
legal=find(score>-1e5);
eps=0; if isfield(W,'fresh_ddqn_epsilon'), eps=double(W.fresh_ddqn_epsilon); end
coverageRate=0;
if isfield(W,'fresh_coverage_rate'), coverageRate=double(W.fresh_coverage_rate); end
coverageMode=isfield(W,'fresh_exploration_mode') && string(W.fresh_exploration_mode)=="coverage";
if coverageMode && coverageRate>0 && numel(legal)>1 && rand()<coverageRate
    ix=coveragePick(W,c,X,legal); action=c(ix); action.detail="fresh_ddqn_coverage";
elseif eps>0 && numel(legal)>1 && rand()<eps
    ix=legal(randi(numel(legal))); action=c(ix); action.detail="fresh_ddqn_explore";
else
    [~,ix]=max(score); action=c(ix); action.detail="fresh_ddqn_greedy";
end

function ix=coveragePick(W,c,X,legal)
counts=zeros(1,800);
if isfield(W,'fresh_coverage_counts') && ~isempty(W.fresh_coverage_counts)
    counts(1:min(numel(counts),numel(W.fresh_coverage_counts)))=W.fresh_coverage_counts(1:min(numel(counts),numel(W.fresh_coverage_counts)));
end
bins=zeros(size(legal));
for i=1:numel(legal), bins(i)=WorkflowFreshCoverageBin(c(legal(i)),X(legal(i),:)); end
w=1./(1+counts(bins)); w=w/sum(w); ix=legal(find(rand()<=cumsum(w),1));
end
end

function score=bcScore(W,X)
score=[];
if ~isfield(W,'fresh_bc_weights_json') || ~isfile(W.fresh_bc_weights_json), return; end
persistent cachePath cache
if isempty(cache) || ~strcmp(char(cachePath),char(W.fresh_bc_weights_json))
    d=jsondecode(fileread(W.fresh_bc_weights_json)); s=d.state_dict;
    if isfield(d,'architecture') && string(d.architecture)=="dueling_ddqn"
        cache=struct('dueling',true, ...
            'ew',double(s.encoder_0_weight),'eb',double(s.encoder_0_bias(:))', ...
            'vw1',double(s.value_0_weight),'vb1',double(s.value_0_bias(:))', ...
            'vw2',double(s.value_2_weight),'vb2',double(s.value_2_bias), ...
            'aw1',double(s.advantage_0_weight),'ab1',double(s.advantage_0_bias(:))', ...
            'aw2',double(s.advantage_2_weight),'ab2',double(s.advantage_2_bias));
    else
        cache=struct('dueling',false,'w1',double(s.net_0_weight),'b1',double(s.net_0_bias(:))', ...
            'w2',double(s.net_2_weight),'b2',double(s.net_2_bias));
    end
    cachePath=W.fresh_bc_weights_json;
end
if cache.dueling
    h=max(0,X*cache.ew'+cache.eb);
    z=mean(h,1);
    vh=max(0,z*cache.vw1'+cache.vb1); v=vh*cache.vw2'+cache.vb2;
    ah=max(0,[h repmat(z,size(h,1),1)]*cache.aw1'+cache.ab1);
    a=ah*cache.aw2'+cache.ab2;
    score=v+a-mean(a);
else
    h=max(0,X*cache.w1'+cache.b1);
    score=h*cache.w2'+cache.b2;
end
end
function tf=categoricalKind(c,k)
tf=false(numel(c),1); for i=1:numel(c), tf(i)=c(i).kind==k; end
end
