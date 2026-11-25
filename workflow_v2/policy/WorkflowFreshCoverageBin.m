function bin=WorkflowFreshCoverageBin(action,x)
%WORKFLOWFRESHCOVERAGEBIN Stable coverage stratum for a legal fresh action.
% The bin is intentionally computed from causal candidate information only.
kind=find(action.kind==["COMMIT_LOCAL" "REQUEST_REMOTE_COMMIT" "RELEASE_EDGE" "REJECT_WORKFLOW" "STOP_PLAN"],1);
if isempty(kind), kind=5; end
stage=find(x(11:14)>0.5,1); if isempty(stage), stage=5; end
helper=1+double(action.helper_id>0);
slack=1+min(3,max(0,floor((max(-1,min(2,x(8)))+1)*(4/3))));
queue=1+min(3,max(0,floor(min(2,max(0,x(6)))*2)));
% 5 kinds x 5 stages x 2 helper groups x 4 slack bins x 4 queue bins.
bin=kind+5*(stage-1)+25*(helper-1)+50*(slack-1)+200*(queue-1);
end
