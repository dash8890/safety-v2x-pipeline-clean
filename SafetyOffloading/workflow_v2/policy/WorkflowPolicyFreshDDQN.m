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
if eps>0 && numel(legal)>1 && rand()<eps
    ix=legal(randi(numel(legal))); action=c(ix); action.detail="fresh_ddqn_explore";
else
    [~,ix]=max(score); action=c(ix); action.detail="fresh_ddqn_greedy";
end
end

function score=bcScore(W,X)
score=[];
if ~isfield(W,'fresh_bc_weights_json') || ~isfile(W.fresh_bc_weights_json), return; end
persistent cachePath cache
if isempty(cache) || ~strcmp(char(cachePath),char(W.fresh_bc_weights_json))
    d=jsondecode(fileread(W.fresh_bc_weights_json)); s=d.state_dict;
    cache=struct('w1',double(s.net_0_weight),'b1',double(s.net_0_bias(:))', ...
        'w2',double(s.net_2_weight),'b2',double(s.net_2_bias));
    cachePath=W.fresh_bc_weights_json;
end
h=max(0,X*cache.w1'+cache.b1);
score=h*cache.w2'+cache.b2;
end
function tf=categoricalKind(c,k)
tf=false(numel(c),1); for i=1:numel(c), tf(i)=c(i).kind==k; end
end
