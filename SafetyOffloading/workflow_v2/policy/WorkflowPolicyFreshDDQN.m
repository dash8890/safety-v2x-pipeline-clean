function [action,X]=WorkflowPolicyFreshDDQN(W)
%WORKFLOWPOLICYFRESHDDQN Candidate-index fresh-action policy entrypoint.
c=WorkflowCandidateBuilder(W); X=WorkflowFreshCandidateFeatures(W,c);
score=bcScore(W,X);
if isempty(score)
    % Safe deterministic bootstrap until BC weights are supplied.
    score=2*X(:,8)-0.15*X(:,6)-0.10*X(:,9)+0.05*X(:,10);
end
score(categoricalKind(c,"STOP_PLAN"))=-1e6;
[~,ix]=max(score); action=c(ix); action.detail="fresh_candidate_bootstrap";
end

function score=bcScore(W,X)
score=[];
if ~isfield(W,'fresh_bc_weights_json') || ~isfile(W.fresh_bc_weights_json), return; end
persistent cachePath cache
if isempty(cache) || cachePath~=W.fresh_bc_weights_json
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
