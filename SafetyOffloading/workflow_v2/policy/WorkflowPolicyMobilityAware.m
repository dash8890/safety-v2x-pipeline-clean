function acts=WorkflowPolicyMobilityAware(W)
%B3: Queue-aware + require OBSERVED contact for remote; prefer reachable helpers.
acts=struct('type',{},'workflow_id',{},'stage_id',{},'helper_id',{},'edge_name',{},'detail',{});
for i=1:numel(W.stages)
    st=W.stages(i);
    if st.status~="ELIGIBLE", continue; end
    [hid,why]=pickMobile(W,st);
    if hid==W.origin_id
        acts(end+1)=mk("COMMIT_LOCAL",st.workflow_id,st.stage_id,0,"",why); %#ok<AGROW>
    else
        acts(end+1)=mk("REQUEST_REMOTE",st.workflow_id,st.stage_id,hid,"",why); %#ok<AGROW>
    end
end
for i=1:numel(W.edges)
    ed=W.edges(i);
    if ed.status~="READY_HELD", continue; end
    if ed.src_vehicle<=0 || ed.dst_vehicle<=0 || ed.src_vehicle==ed.dst_vehicle, continue; end
    if W.contact(ed.src_vehicle,ed.dst_vehicle)
        acts(end+1)=mk("RELEASE_EDGE",ed.workflow_id,"",0,ed.name,"contact_ok"); %#ok<AGROW>
    end
end
end

function [hid,why]=pickMobile(W,st)
best=inf; hid=W.origin_id; why="local_mob";
for v=1:numel(W.cpu)
    if v~=W.origin_id && ~W.contact(W.origin_id,v)
        continue % unreachable helper
    end
    q=0;
    if ~isempty(W.cpu(v).running), q=q+W.cpu(v).running.remaining; end
    if ~isempty(W.cpu(v).queue)
        for j=1:numel(W.cpu(v).queue), q=q+W.cpu(v).queue(j).remaining; end
    end
    t=q/max(W.cpu(v).available_hz,1)+st.required_cycles/max(W.cpu(v).available_hz,1);
    if v~=W.origin_id, t=t+2*max(W.default_link_delay_s,0.010); end
    if t<best, best=t; hid=v; why=sprintf('mob_v%d_t=%.4f',v,t); end
end
end

function a=mk(type,wf,sid,hid,ename,detail)
a=struct('type',string(type),'workflow_id',wf,'stage_id',string(sid), ...
    'helper_id',hid,'edge_name',string(ename),'detail',string(detail));
end
