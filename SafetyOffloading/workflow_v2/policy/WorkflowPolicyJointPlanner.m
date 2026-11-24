function acts=WorkflowPolicyJointPlanner(W)
%B4: Deployable joint planner. Helper state is read only from fresh received
% HELPER_ADV messages; true helper queues are reserved for B5 diagnostics.
acts=struct('type',{},'workflow_id',{},'stage_id',{},'helper_id',{},'edge_name',{},'detail',{});
% Placement
for i=1:numel(W.stages)
    st=W.stages(i);
    if st.status~="ELIGIBLE", continue; end
    sid=string(st.stage_id);
    if sid=="A" || sid=="D"
        acts(end+1)=mk("COMMIT_LOCAL",st.workflow_id,sid,0,"","joint_local_AD"); %#ok<AGROW>
        continue
    end
    % B/C: prefer distinct helpers if both reachable and faster than local
    [hid,why]=pickJoint(W,st,sid,acts);
    if hid==W.origin_id
        acts(end+1)=mk("COMMIT_LOCAL",st.workflow_id,sid,0,"",why); %#ok<AGROW>
    else
        acts(end+1)=mk("REQUEST_REMOTE",st.workflow_id,sid,hid,"",why); %#ok<AGROW>
    end
end
% Controlled release: release READY_HELD only if contact and dest committed
for i=1:numel(W.edges)
    ed=W.edges(i);
    if ed.status~="READY_HELD", continue; end
    if ed.src_vehicle<=0 || ed.dst_vehicle<=0, continue; end
    if ed.src_vehicle==ed.dst_vehicle, continue; end % local auto
    if ~W.contact(ed.src_vehicle,ed.dst_vehicle), continue; end
    % Slack: hold if deadline very tight and another edge still pending? simple: release
    acts(end+1)=mk("RELEASE_EDGE",ed.workflow_id,"",0,ed.name,"joint_release"); %#ok<AGROW>
end
end

function [hid,why]=pickJoint(W,st,sid,priorActs)
% Prefer distinct advertised helpers for B/C.  Origin queue is local state;
% helper queue/capacity comes only from a fresh received advertisement.
used=[];
for i=1:numel(priorActs)
    if priorActs(i).type=="REQUEST_REMOTE", used(end+1)=priorActs(i).helper_id; %#ok<AGROW>
    end
end
best=inf; hid=W.origin_id; why="joint_local";
localT=queued(W.cpu(W.origin_id))/max(W.cpu(W.origin_id).available_hz,1) ...
    +st.required_cycles/max(W.cpu(W.origin_id).available_hz,1);
for v=1:numel(W.cpu)
    if v==W.origin_id, continue; end
    if ismember(v,used), continue; end
    a=receivedAdv(W,v); if isempty(a) || ~a.willingness, continue; end
    t=(a.queued_cycles+a.admitted_pending_cycles+st.required_cycles)/ ...
        max(a.available_cpu_hz,1)+0.050;
    % Bias toward remote when helper hz is higher (joint opportunity).
    if a.available_cpu_hz>W.cpu(W.origin_id).available_hz
        t=t*0.85;
    end
    if t<best, best=t; hid=v; why=sprintf('joint_%s_v%d',sid,v); end
end

if hid~=W.origin_id && best>localT*1.5
    hid=W.origin_id; why="joint_local_fallback";
end
end

function a=receivedAdv(W,helper)
a=[]; newest=-inf;
if ~isfield(W,'advertisements'), return; end
for i=1:numel(W.advertisements)
    x=W.advertisements(i);
    if x.helper_id~=helper || W.now_s>x.rx_s+x.ttl_s+1e-12, continue; end
    if x.rx_s>newest, a=x; newest=x.rx_s; end
end
end

function q=queued(S)
q=0;
if ~isempty(S.running), q=q+S.running.remaining; end
if ~isempty(S.queue)
    for j=1:numel(S.queue), q=q+S.queue(j).remaining; end
end
end

function a=mk(type,wf,sid,hid,ename,detail)
a=struct('type',string(type),'workflow_id',wf,'stage_id',string(sid), ...
    'helper_id',hid,'edge_name',string(ename),'detail',string(detail));
end
