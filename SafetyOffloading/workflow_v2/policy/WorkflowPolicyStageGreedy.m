function acts=WorkflowPolicyStageGreedy(W)
%B1: Place each eligible stage on min estimated completion time; immediate release.
acts=struct('type',{},'workflow_id',{},'stage_id',{},'helper_id',{},'edge_name',{},'detail',{});
for i=1:numel(W.stages)
    st=W.stages(i);
    if st.status~="ELIGIBLE", continue; end
    [hid,why]=pickMinFinish(W,st);
    if hid==W.origin_id
        acts(end+1)=mk("COMMIT_LOCAL",st.workflow_id,st.stage_id,0,"",why); %#ok<AGROW>
    else
        acts(end+1)=mk("REQUEST_REMOTE",st.workflow_id,st.stage_id,hid,"",why); %#ok<AGROW>
    end
end
% Immediate release of READY_HELD remote edges
for i=1:numel(W.edges)
    ed=W.edges(i);
    if ed.status=="READY_HELD" && ed.src_vehicle~=ed.dst_vehicle && ed.dst_vehicle>0
        acts(end+1)=mk("RELEASE_EDGE",ed.workflow_id,"",0,ed.name,"immediate"); %#ok<AGROW>
    end
end
end

function [hid,why]=pickMinFinish(W,st)
% Stateless, deployable placement: own CPU is local state; helper capacity
% is used only after a received HELPER_ADV.  No helper queue is inspected.
best=st.required_cycles/max(W.cpu(W.origin_id).available_hz,1);
hid=W.origin_id; why="local";
for v=1:numel(W.cpu)
    if v==W.origin_id, continue; end
    a=receivedAdv(W,v); if isempty(a) || ~a.willingness, continue; end
    t=st.required_cycles/max(a.available_cpu_hz,1)+2*0.025;
    if t<best, best=t; hid=v; why=sprintf('adv_greedy_v%d_t=%.4f',v,t); end
end
end

function a=receivedAdv(W,helper)
a=[]; best=-inf;
if ~isfield(W,'advertisements'), return; end
for i=1:numel(W.advertisements)
    x=W.advertisements(i);
    if x.helper_id~=helper || W.now_s>x.rx_s+x.ttl_s+1e-12, continue; end
    if x.rx_s>best, a=x; best=x.rx_s; end
end
end

function a=mk(type,wf,sid,hid,ename,detail)
a=struct('type',string(type),'workflow_id',wf,'stage_id',string(sid), ...
    'helper_id',hid,'edge_name',string(ename),'detail',string(detail));
end
