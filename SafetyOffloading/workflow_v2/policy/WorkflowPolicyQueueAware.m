function acts=WorkflowPolicyQueueAware(W)
%B2: Placement accounts for ego/helper queued+running cycles; immediate release.
acts=struct('type',{},'workflow_id',{},'stage_id',{},'helper_id',{},'edge_name',{},'detail',{});
for i=1:numel(W.stages)
    st=W.stages(i);
    if st.status~="ELIGIBLE", continue; end
    [hid,why]=pickQueueAware(W,st);
    if hid==W.origin_id
        acts(end+1)=mk("COMMIT_LOCAL",st.workflow_id,st.stage_id,0,"",why); %#ok<AGROW>
    else
        acts(end+1)=mk("REQUEST_REMOTE",st.workflow_id,st.stage_id,hid,"",why); %#ok<AGROW>
    end
end
for i=1:numel(W.edges)
    ed=W.edges(i);
    if ed.status=="READY_HELD" && ed.src_vehicle>0 && ed.dst_vehicle>0 && ed.src_vehicle~=ed.dst_vehicle
        acts(end+1)=mk("RELEASE_EDGE",ed.workflow_id,"",0,ed.name,"immediate"); %#ok<AGROW>
    end
end
end

function [hid,why]=pickQueueAware(W,st)
best=inf; hid=W.origin_id; why="local_q";
for v=1:numel(W.cpu)
    q=queuedCycles(W.cpu(v));
    tWait=q/max(W.cpu(v).available_hz,1);
    tComp=st.required_cycles/max(W.cpu(v).available_hz,1);
    tLink=0; if v~=W.origin_id, tLink=2*max(W.default_link_delay_s,0.010); end
    t=tWait+tComp+tLink;
    if t<best, best=t; hid=v; why=sprintf('queue_v%d_t=%.4f_q=%g',v,t,q); end
end
end

function q=queuedCycles(S)
q=0;
if ~isempty(S.running), q=q+S.running.remaining; end
if ~isempty(S.queue)
    for i=1:numel(S.queue), q=q+S.queue(i).remaining; end
end
end

function a=mk(type,wf,sid,hid,ename,detail)
a=struct('type',string(type),'workflow_id',wf,'stage_id',string(sid), ...
    'helper_id',hid,'edge_name',string(ename),'detail',string(detail));
end
