function X=WorkflowFreshCandidateFeatures(W,candidates)
%WORKFLOWFRESHCANDIDATEFEATURES Causal features for fresh action candidates.
% Rows align exactly with WorkflowCandidateBuilder's executable action union.
n=numel(candidates); X=zeros(n,16);
originQ=queued(W.cpu(W.origin_id)); originHz=max(W.cpu(W.origin_id).available_hz,1);
n_data_frags=0;
if isfield(W,'log')
    for li=1:numel(W.log)
        e=W.log(li);
        if (e.type=="FRAGMENT_TX"||e.type=="FRAGMENT_RX") && ~contains(string(e.detail),"ctrl")
            n_data_frags=n_data_frags+1;
        end
    end
end
frag_rate=min(1,n_data_frags/200);
for i=1:n
    a=candidates(i); X(i,1:5)=double(a.kind==["COMMIT_LOCAL" "REQUEST_REMOTE_COMMIT" "RELEASE_EDGE" "REJECT_WORKFLOW" "STOP_PLAN"]);
    X(i,6)=min(2,originQ/(originHz*max(W.rel_deadline_s,1e-9)));
    [cycles,slack]=stageState(W,a); X(i,7)=cycles/1e7; X(i,8)=max(-1,min(2,slack/max(W.rel_deadline_s,1e-9)));
    if a.helper_id>0
        adv=latestAdv(W,a.helper_id);
        if ~isempty(adv)
            X(i,9)=min(2,(adv.queued_cycles+adv.admitted_pending_cycles)/max(adv.available_cpu_hz*W.rel_deadline_s,1));
            X(i,10)=adv.available_cpu_hz/max([W.cpu.available_hz]);
        end
    end
    X(i,11:14)=stageOneHot(a.stage_id);
    X(i,15)=a.helper_id/max(numel(W.cpu),1);
    X(i,16)=frag_rate;
end

function z=stageOneHot(stage)
z=zeros(1,4);
switch string(stage)
    case "A", z(1)=1; case "B", z(2)=1; case "C", z(3)=1; case "D", z(4)=1;
end
end
end
function q=queued(S)
q=0; if ~isempty(S.running), q=q+S.running.remaining; end
for j=1:numel(S.queue), q=q+S.queue(j).remaining; end
end
function [cycles,slack]=stageState(W,a)
cycles=0; slack=0;
for i=1:numel(W.stages)
    s=W.stages(i); if s.workflow_id==a.workflow_id && s.stage_id==a.stage_id
        cycles=s.required_cycles; for j=1:numel(W.workflows), if W.workflows(j).workflow_id==a.workflow_id, slack=W.workflows(j).absolute_deadline_s-W.now_s; end, end; return
    end
end
end
function a=latestAdv(W,h)
a=[]; t=-inf; if ~isfield(W,'advertisements'), return; end
for i=1:numel(W.advertisements)
    x=W.advertisements(i); if x.helper_id==h && W.now_s<=x.rx_s+x.ttl_s && x.rx_s>t, a=x; t=x.rx_s; end
end
end
