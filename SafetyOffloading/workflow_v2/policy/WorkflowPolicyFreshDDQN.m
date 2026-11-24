function [action,X]=WorkflowPolicyFreshDDQN(W)
%WORKFLOWPOLICYFRESHDDQN Candidate-index fresh-action policy entrypoint.
c=WorkflowCandidateBuilder(W); X=WorkflowFreshCandidateFeatures(W,c);
% Safe deterministic bootstrap until BC weights are supplied: maximize
% deadline slack while preferring remote candidates only when their received
% advertisement indicates lower normalized queue pressure.
score=2*X(:,8)-0.15*X(:,6)-0.10*X(:,9)+0.05*X(:,10);
score(categoricalKind(c,"STOP_PLAN"))=-1e6;
[~,ix]=max(score); action=c(ix); action.detail="fresh_candidate_bootstrap";
end
function tf=categoricalKind(c,k)
tf=false(numel(c),1); for i=1:numel(c), tf(i)=c(i).kind==k; end
end
