function acts=WorkflowPolicyDropLow(W)
%PDP Priority-Based Demonstration Policy (Always-Drop-Low).
% BC teacher only — not an optimization oracle, not a literature baseline.
acts=struct('type',{},'workflow_id',{},'stage_id',{},'helper_id',{},'edge_name',{},'detail',{});
st=pickEligibleEDF(W);
if isempty(st)
    acts=releaseReadyEdges(W);
    return
end
wf=wfOf(W,st.workflow_id);
vh=12; if isfield(W,'value_high'), vh=double(W.value_high); end
if wf.value<vh-1e-9 && st.stage_id=="A"
    acts(1)=mk("DROP",st.workflow_id,st.stage_id,0,"","drop_low");
    return
end
if st.stage_id=="A" || st.stage_id=="D"
    acts(1)=mk("COMMIT_LOCAL",st.workflow_id,st.stage_id,0,"","anchor_local");
    return
end
[hid,why]=pickMobile(W,st);
if hid==W.origin_id
    acts(1)=mk("COMMIT_LOCAL",st.workflow_id,st.stage_id,0,"",why);
else
    acts(1)=mk("REQUEST_REMOTE",st.workflow_id,st.stage_id,hid,"",why);
end
end

function st=pickEligibleEDF(W)
st=[];
best=inf;
for i=1:numel(W.stages)
    s=W.stages(i);
    if s.status~="ELIGIBLE", continue; end
    wf=wfOf(W,s.workflow_id);
    key=wf.absolute_deadline_s+1e-9*double(s.workflow_id);
    if key<best, best=key; st=s; end
end
end

function acts=releaseReadyEdges(W)
acts=struct('type',{},'workflow_id',{},'stage_id',{},'helper_id',{},'edge_name',{},'detail',{});
for i=1:numel(W.edges)
    ed=W.edges(i);
    if ed.status~="READY_HELD", continue; end
    if ed.src_vehicle<=0 || ed.dst_vehicle<=0 || ed.src_vehicle==ed.dst_vehicle, continue; end
    if W.contact(ed.src_vehicle,ed.dst_vehicle)
        acts(end+1)=mk("RELEASE_EDGE",ed.workflow_id,"",0,ed.name,"contact_ok"); %#ok<AGROW>
    end
end
end

function wf=wfOf(W,id)
wf=W.workflows(1);
for i=1:numel(W.workflows)
    if W.workflows(i).workflow_id==id, wf=W.workflows(i); return; end
end
end

function [hid,why]=pickMobile(W,st)
best=inf; hid=W.origin_id; why="local_mob";
for v=1:numel(W.cpu)
    if v==W.origin_id
        q=queued(W.cpu(v)); rate=W.cpu(v).available_hz; link=0;
    else
        a=receivedAdv(W,v); if isempty(a) || ~a.willingness, continue; end
        q=a.queued_cycles+a.admitted_pending_cycles; rate=a.available_cpu_hz; link=0.050;
    end
    t=(q+st.required_cycles)/max(rate,1)+link;
    if t<best, best=t; hid=v; why=sprintf('mob_v%d_t=%.4f',v,t); end
end
end

function q=queued(S)
q=0; if ~isempty(S.running), q=q+S.running.remaining; end
for i=1:numel(S.queue), q=q+S.queue(i).remaining; end
end

function a=receivedAdv(W,helper)
a=[]; newest=-inf; if ~isfield(W,'advertisements'), return; end
for i=1:numel(W.advertisements)
    x=W.advertisements(i);
    if x.helper_id~=helper || W.now_s>x.rx_s+x.ttl_s+1e-12, continue; end
    if x.rx_s>newest, a=x; newest=x.rx_s; end
end
end

function a=mk(type,wf,sid,hid,ename,detail)
a=struct('type',string(type),'workflow_id',wf,'stage_id',string(sid), ...
    'helper_id',hid,'edge_name',string(ename),'detail',string(detail));
end
