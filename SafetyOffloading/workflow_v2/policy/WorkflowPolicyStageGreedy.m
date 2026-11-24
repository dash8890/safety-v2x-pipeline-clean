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
% Ignore queues (independent placement): service = cycles/hz + crude link.
best=inf; hid=W.origin_id; why="local";
for v=1:numel(W.cpu)
    tComp=st.required_cycles/max(W.cpu(v).available_hz,1);
    tLink=0;
    if v~=W.origin_id, tLink=2*W.default_link_delay_s; end
    t=tComp+tLink;
    if t<best, best=t; hid=v; why=sprintf('greedy_v%d_t=%.4f',v,t); end
end
end

function a=mk(type,wf,sid,hid,ename,detail)
a=struct('type',string(type),'workflow_id',wf,'stage_id',string(sid), ...
    'helper_id',hid,'edge_name',string(ename),'detail',string(detail));
end
