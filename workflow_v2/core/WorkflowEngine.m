function varargout=WorkflowEngine(op,varargin)
%WORKFLOWENGINE Diamond-4 workflow lifecycle for R2 deterministic fixtures.
% Uses frozen R1 adapters. Radio path is scripted_unit (labeled); no free
% helper oracle — remote commit still goes through WorkflowControlProtocol.
switch string(op)
    case "new", varargout{1}=newWorld(varargin{:});
    case "create_workflow", [varargout{1},varargout{2}]=createWorkflow(varargin{:});
    case "commit_local", varargout{1}=commitLocal(varargin{:});
    case "request_remote", varargout{1}=requestRemote(varargin{:});
    case "release_edge", varargout{1}=releaseEdge(varargin{:});
    case "set_contact", varargout{1}=setContact(varargin{:});
    case "set_link_delay_ms", varargout{1}=setLinkDelay(varargin{:});
    case "enqueue_background", varargout{1}=enqueueBackground(varargin{:});
    case "advance_to", varargout{1}=advanceTo(varargin{:});
    case "inject_ctrl_rx", varargout{1}=injectCtrlRx(varargin{:});
    case "inject_fragment_rx", varargout{1}=injectFragmentRx(varargin{:});
    case "repeat_ack", varargout{1}=repeatAck(varargin{:});
    case "repeat_fragment", varargout{1}=repeatFragment(varargin{:});
    case "summary", varargout{1}=WorkflowMetrics('summary',varargin{1});
    case "find_stage", [varargout{1},varargout{2}]=findStage(varargin{:});
    case "find_edge", [varargout{1},varargout{2}]=findEdge(varargin{:});
    case "__on_cpu_complete", varargout{1}=onCpuComplete(varargin{:});
    case "__check_success", varargout{1}=checkWorkflowSuccess(varargin{:});
    case "__expire", varargout{1}=expireWorkflows(varargin{:});
    case "__handle_ctrl", varargout{1}=handleCtrl(varargin{:});
    case "__handle_fragment", varargout{1}=handleFragment(varargin{:});
    case "auto_release_ready", varargout{1}=autoReleaseReady(varargin{:});
    case "drop_workflow", varargout{1}=dropWorkflow(varargin{:});
    otherwise, error('WorkflowEngine:UnknownOp','Unknown op %s',op);
end
end

function W=newWorld(cfg)
if nargin<1, cfg=struct(); end
W=struct();
W.now_s=0;
W.origin_id=getf(cfg,'origin_id',1);
W.n_vehicles=getf(cfg,'n_vehicles',3);
W.tb_bytes=getf(cfg,'tb_bytes',1180);
W.rel_deadline_s=getf(cfg,'rel_deadline_s',0.5);
W.stage_cycles=getf(cfg,'stage_cycles',[9e6 12e6 12e6 3e6]);
assert(numel(W.stage_cycles)==4 && all(W.stage_cycles>0), ...
    'WorkflowEngine:StageCycles','stage_cycles must contain four positive values');
W.ctrl_delay_s=getf(cfg,'ctrl_delay_s',0.005);
W.default_link_delay_s=getf(cfg,'default_link_delay_s',0.010);
W.slot_s=getf(cfg,'slot_s',0.0005);
W.mode="scripted_unit";
W.next_wf=1; W.next_edge=1; W.next_msg=1; W.next_job=1; W.next_nonce=1;
W.workflows=struct('workflow_id',{},'origin_id',{},'release_s',{}, ...
    'absolute_deadline_s',{},'value',{},'topology_id',{},'status',{}, ...
    'completion_s',{},'terminal_stage',"D");
W.stages=struct('workflow_id',{},'stage_id',{},'required_cycles',{}, ...
    'status',{},'placement_id',{},'commitment_id',{},'commitment_acked',{}, ...
    'remaining_cycles',{},'start_s',{},'finish_s',{},'pred_edges',{}, ...
    'succ_edges',{},'remote_attempts',{},'req_nonce',{},'req_sent_ms',{}, ...
    'job_id',{},'last_nack',{},'late_ack',{},'commit_ack_ms',{});
W.edges=struct('edge_id',{},'workflow_id',{},'name',{},'src_stage',{}, ...
    'dst_stage',{},'size_bytes',{},'src_vehicle',{},'dst_vehicle',{}, ...
    'status',{},'release_s',{},'fully_received_s',{},'fragment_ids',{}, ...
    'planned_frags',{},'planned_fragments',{},'commitment_acked',{}, ...
    'commitment_id',{},'message_id',{},'rx_count',{},'mac_enqueued',{}, ...
    'held',{},'deadline_ms',{},'tb_bytes_assumed',{});
hz=getf(cfg,'cpu_hz',[5e8 1.5e9 2e9]);
W.cpu=repmat(WorkflowCpuQueue('new',1,hz(1),0),1,W.n_vehicles);
for v=1:W.n_vehicles
    W.cpu(v)=WorkflowCpuQueue('new',v,hz(min(v,numel(hz))),0);
end
W.ledger=repmat({WorkflowControlProtocol('empty_ledger')},1,W.n_vehicles);
W.origin_ledger=WorkflowControlProtocol('empty_ledger');
W.pending_tx=repmat({[]},1,W.n_vehicles);
W.contact=true(W.n_vehicles); % fully connected by default
W.link_delay=containers.Map('KeyType','char','ValueType','double');
W.event_q={}; % cell of structs — payloads differ by kind
W.log=WorkflowMetrics('empty');
W.buffered_result=struct('workflow_id',{},'stage_id',{},'edge_name',{}, ...
    'helper_id',{},'ready_s',{});
W.ack_cache=struct('nonce',{},'ack',{},'stage_key',{});
W.frag_rx_seen=containers.Map('KeyType','char','ValueType','logical');
W.max_admitted_cycles=getf(cfg,'max_admitted_cycles',2*5e8*0.5);
W.willingness=true(1,W.n_vehicles);
W.last_d_start_s=NaN;
W.advertisements=struct('helper_id',{},'rx_s',{},'generated_s',{},'ttl_s',{}, ...
    'nominal_cpu_hz',{},'available_cpu_hz',{},'queued_cycles',{}, ...
    'running_remaining_cycles',{},'admitted_pending_cycles',{},'willingness',{},'seq',{});
end

function [W,wfId]=createWorkflow(W,releaseS)
if nargin<2, releaseS=W.now_s; end
wfId=W.next_wf; W.next_wf=W.next_wf+1;
deadline=releaseS+W.rel_deadline_s;
val=assignWorkflowValue(W,releaseS,wfId);
wf=struct('workflow_id',wfId,'origin_id',W.origin_id,'release_s',releaseS, ...
    'absolute_deadline_s',deadline,'value',val,'topology_id',"diamond4", ...
    'status',"ACTIVE",'completion_s',NaN,'terminal_stage',"D");
W.workflows(end+1)=wf; %#ok<AGROW>

% Stages A/B/C/D
specs={{"A",W.stage_cycles(1),{},{}},{"B",W.stage_cycles(2),{},{}}, ...
       {"C",W.stage_cycles(3),{},{}},{"D",W.stage_cycles(4),{},{}}};
for i=1:numel(specs)
    st=emptyStage(wfId,specs{i}{1},specs{i}{2});
    W.stages(end+1)=st; %#ok<AGROW>
end

% Edges
edgesDef={
    "root_A","", "A", 1500
    "A_B",   "A","B", 900
    "A_C",   "A","C", 900
    "B_D",   "B","D", 600
    "C_D",   "C","D", 600
    "D_return","D","", 300
    };
for i=1:size(edgesDef,1)
    [W,eid]=addEdge(W,wfId,edgesDef{i,1},edgesDef{i,2},edgesDef{i,3},edgesDef{i,4},deadline);
    % wire preds/succs
    if edgesDef{i,3}~=""
        [~,si]=findStage(W,wfId,edgesDef{i,3});
        W.stages(si).pred_edges(end+1)=eid; %#ok<AGROW>
    end
    if edgesDef{i,2}~=""
        [~,si]=findStage(W,wfId,edgesDef{i,2});
        W.stages(si).succ_edges(end+1)=eid; %#ok<AGROW>
    end
end

% Root input already at origin
[~,ei]=findEdge(W,wfId,"root_A");
W.edges(ei).src_vehicle=W.origin_id;
W.edges(ei).dst_vehicle=W.origin_id;
W.edges(ei).status="DELIVERED";
W.edges(ei).fully_received_s=releaseS;
W.edges(ei).commitment_acked=true;

W=refreshEligibility(W,wfId);
W.log=WorkflowMetrics('log',W.log,releaseS,"WORKFLOW_ARRIVAL",'workflow_id',wfId);
W=scheduleEvent(W,releaseS,"CONTROLLER",struct('workflow_id',wfId));
end

function st=emptyStage(wfId,sid,cycles)
st=struct('workflow_id',wfId,'stage_id',string(sid),'required_cycles',cycles, ...
    'status',"BLOCKED",'placement_id',0,'commitment_id',0,'commitment_acked',false, ...
    'remaining_cycles',cycles,'start_s',NaN,'finish_s',NaN,'pred_edges',[], ...
    'succ_edges',[],'remote_attempts',0,'req_nonce',0,'req_sent_ms',NaN, ...
    'job_id',0,'last_nack',"",'late_ack',false,'commit_ack_ms',NaN);
end

function [W,eid]=addEdge(W,wfId,name,srcSt,dstSt,bytes,deadlineS)
eid=W.next_edge; W.next_edge=W.next_edge+1;
mid=W.next_msg; W.next_msg=W.next_msg+1;
e=struct('edge_id',eid,'workflow_id',wfId,'name',string(name), ...
    'src_stage',string(srcSt),'dst_stage',string(dstSt),'size_bytes',bytes, ...
    'src_vehicle',0,'dst_vehicle',0,'status',"NOT_MATERIALIZED", ...
    'release_s',NaN,'fully_received_s',NaN,'fragment_ids',[], ...
    'planned_frags',[],'planned_fragments',0,'commitment_acked',false, ...
    'commitment_id',0,'message_id',mid,'rx_count',0,'mac_enqueued',false, ...
    'held',false,'deadline_ms',deadlineS*1000,'tb_bytes_assumed',W.tb_bytes);
W.edges(end+1)=e; %#ok<AGROW>
end

function W=commitLocal(W,wfId,stageId)
[st,si]=findStage(W,wfId,stageId);
assert(st.status=="ELIGIBLE" || st.status=="BLOCKED" || st.status=="DATA_WAIT" ...
    || st.status=="COMMITTED",'bad stage status for local commit');
% Disallow early commit: predecessors must be DONE (root ok via delivered input)
assert(predsDone(W,st),'WorkflowEngine:EarlyCommit','predecessors not DONE');
st.status="COMMITTED";
st.placement_id=W.origin_id;
st.commitment_acked=true;
W.stages(si)=st;
W.log=WorkflowMetrics('log',W.log,W.now_s,"COMMIT_LOCAL",'workflow_id',wfId, ...
    'stage_id',string(stageId),'vehicle_id',W.origin_id);
W=autoReleaseReady(W,wfId);
W=tryEnqueueStage(W,wfId,stageId);
end

function W=requestRemote(W,wfId,stageId,helperId)
[st,si]=findStage(W,wfId,stageId);
assert(predsDone(W,st),'WorkflowEngine:EarlyCommit');
assert(helperId~=W.origin_id,'use commit_local for origin');
inBytes=inputBytesForStage(W,st);
nonce=W.next_nonce; W.next_nonce=W.next_nonce+1;
nowMs=W.now_s*1000;
req=WorkflowControlProtocol('make_req',wfId,string(stageId),helperId, ...
    st.required_cycles,W.workflows(wfIndex(W,wfId)).absolute_deadline_s*1000, ...
    inBytes,nonce,nowMs,W.origin_id);
[W.pending_tx,meta]=WorkflowControlProtocol('enqueue_ctrl',W.pending_tx,W.origin_id,req,W.tb_bytes); %#ok<ASGLU>
W.log=WorkflowMetrics('log',W.log,W.now_s,"FRAGMENT_TX",'workflow_id',wfId, ...
    'stage_id',string(stageId),'vehicle_id',W.origin_id,'detail',"ctrl COMMIT_REQ");
st.status="ADMISSION_PENDING";
st.placement_id=helperId;
st.req_nonce=nonce;
st.req_sent_ms=nowMs;
W.stages(si)=st;
if ~contactOk(W,W.origin_id,helperId)
    W.log=WorkflowMetrics('log',W.log,W.now_s,"CTRL_BLOCKED_NO_CONTACT", ...
        'workflow_id',wfId,'stage_id',string(stageId),'vehicle_id',helperId);
elseif isfield(W,'mode') && W.mode=="wilab_mode2"
    % Air delivery via WiLab Mode-2 (bridge drains pending_tx).
elseif contactOk(W,W.origin_id,helperId)
    % scripted_unit / abstract_link
    W=scheduleEvent(W,W.now_s+W.ctrl_delay_s,"CTRL_DELIVER",struct( ...
        'msg',req,'src',W.origin_id,'dst',helperId,'meta',meta));
end
end

function W=releaseEdge(W,wfId,edgeName)
[ed,ei]=findEdge(W,wfId,edgeName);
assert(ed.status=="READY_HELD" || ed.status=="WAIT_ADMISSION", ...
    'edge not releasable');
% Resolve vehicles from stage placements
ed=resolveEdgeVehicles(W,ed);
if ed.src_vehicle==ed.dst_vehicle
    [ed,~]=WorkflowMode2Adapter('plan_fragments',ed,W.tb_bytes,W.now_s*1000);
    [ed,~]=WorkflowMode2Adapter('release_local',ed,W.now_s*1000);
    W.edges(ei)=ed;
    W.log=WorkflowMetrics('log',W.log,W.now_s,"EDGE_LOCAL_DELIVER", ...
        'workflow_id',wfId,'edge_id',ed.edge_id,'detail',string(edgeName));
    W=onEdgeDelivered(W,wfId,edgeName);
    return
end
% Remote: need commitment on destination stage (or return-to-origin).
dstSt=ed.dst_stage;
if dstSt~=""
    [st,~]=findStage(W,wfId,dstSt);
    if ~(st.commitment_acked || st.placement_id==ed.src_vehicle)
        W.log=WorkflowMetrics('log',W.log,W.now_s,"RELEASE_WAIT_COMMIT", ...
            'workflow_id',wfId,'edge_id',ed.edge_id,'detail',string(edgeName));
        return
    end
    ed.commitment_acked=st.commitment_acked;
    ed.commitment_id=st.commitment_id;
end
[ed,~]=WorkflowMode2Adapter('plan_fragments',ed,W.tb_bytes,W.now_s*1000);
ed.commitment_acked=true;
if ~contactOk(W,ed.src_vehicle,ed.dst_vehicle)
    % Buffer: keep READY_HELD / WAIT — do not enqueue
    ed.status="READY_HELD";
    ed.held=true;
    W.edges(ei)=ed;
    W.log=WorkflowMetrics('log',W.log,W.now_s,"TX_BLOCKED_NO_CONTACT", ...
        'workflow_id',wfId,'edge_id',ed.edge_id,'detail',string(edgeName));
    return
end
[ed,W.pending_tx]=WorkflowMode2Adapter('release',ed,W.pending_tx,ed.src_vehicle);
W.edges(ei)=ed;
nFrag=ed.planned_fragments;
W.log=WorkflowMetrics('log',W.log,W.now_s,"EDGE_RELEASE",'workflow_id',wfId, ...
    'edge_id',ed.edge_id,'detail',sprintf('%s frags=%d',edgeName,nFrag));
for f=1:nFrag
    W.log=WorkflowMetrics('log',W.log,W.now_s,"FRAGMENT_TX",'workflow_id',wfId, ...
        'edge_id',ed.edge_id,'vehicle_id',ed.src_vehicle, ...
        'detail',sprintf('data f=%d/%d',f,nFrag));
end
if isfield(W,'mode') && W.mode=="wilab_mode2"
    % Fragments sit on pending_tx until WiLab Mode-2 delivers (bridge RX inject).
    return
end
% scripted_unit / abstract_link: schedule synthetic delivery
delay=linkDelay(W,ed.src_vehicle,ed.dst_vehicle,edgeName);
for f=1:nFrag
    frag=ed.planned_frags(f);
    W=scheduleEvent(W,W.now_s+delay,"FRAGMENT_DELIVER",struct( ...
        'workflow_id',wfId,'edge_name',string(edgeName),'fragment',f, ...
        'fragments',nFrag,'message_id',ed.message_id,'src',ed.src_vehicle, ...
        'dst',ed.dst_vehicle,'bytes',frag.bytes));
end
end

function W=setContact(W,a,b,tf)
W.contact(a,b)=logical(tf);
W.contact(b,a)=logical(tf);
W.log=WorkflowMetrics('log',W.log,W.now_s,"CONTACT",'vehicle_id',a, ...
    'detail',sprintf('%d<->%d = %d',a,b,tf));
% If contact restored, try flush buffered results / held edges
if tf
    W=flushBuffered(W);
end
end

function W=setLinkDelay(W,src,dst,edgeName,delayMs)
key=sprintf('%d_%d_%s',src,dst,edgeName);
W.link_delay(key)=delayMs/1000;
end

function W=enqueueBackground(W,vehicleId,cycles,deadlineS)
jid=W.next_job; W.next_job=W.next_job+1;
meta=struct('kind',"background");
W.cpu(vehicleId)=WorkflowCpuQueue('enqueue',W.cpu(vehicleId),jid,"background", ...
    cycles,deadlineS,W.now_s,meta);
W.log=WorkflowMetrics('log',W.log,W.now_s,"BG_ENQUEUE",'vehicle_id',vehicleId, ...
    'detail',sprintf('cycles=%g job=%d',cycles,jid));
end

function W=advanceTo(W,tEnd)
if tEnd+1e-12<W.now_s
    return
end
while true
    % Next event time or tEnd
    nextT=tEnd;
    if ~isempty(W.event_q)
        ts=cellfun(@(e)e.t_s,W.event_q);
        nextT=min(nextT,min(ts));
    end
    % Also next CPU completion
    for v=1:numel(W.cpu)
        ct=cpuCompletionTime(W.cpu(v));
        if isfinite(ct), nextT=min(nextT,ct); end
    end
    if nextT>tEnd+1e-15, nextT=tEnd; end
    W=WorkflowEventDispatcher('step_to',W,nextT);
    if abs(W.now_s-tEnd)<1e-12 || W.now_s>=tEnd-1e-15
        if W.now_s<tEnd-1e-15
            W=WorkflowEventDispatcher('step_to',W,tEnd);
        end
        break
    end
    if isempty(W.event_q)
        % only CPU progress left
        anyRun=false;
        for v=1:numel(W.cpu)
            if ~isempty(W.cpu(v).running), anyRun=true; break; end
        end
        if ~anyRun && W.now_s>=tEnd-1e-15, break; end
        if ~anyRun
            W=WorkflowEventDispatcher('step_to',W,tEnd);
            break
        end
    end
end
end

%% ---- internal helpers (same file) ----

function W=scheduleEvent(W,t,kind,payload)
W.event_q{end+1}=struct('t_s',t,'kind',string(kind),'payload',payload); %#ok<AGROW>
end

function tf=contactOk(W,a,b)
if a==b, tf=true; return; end
tf=W.contact(a,b);
end

function d=linkDelay(W,src,dst,edgeName)
key=sprintf('%d_%d_%s',src,dst,edgeName);
if isKey(W.link_delay,key)
    d=W.link_delay(key);
else
    d=W.default_link_delay_s;
end
end

function tf=predsDone(W,st)
tf=true;
for eid=st.pred_edges
    [ed,~]=findEdgeById(W,eid);
    if ed.src_stage==""
        % root input: need DELIVERED
        if ed.status~="DELIVERED", tf=false; return; end
    else
        [pst,~]=findStage(W,st.workflow_id,ed.src_stage);
        if pst.status~="DONE", tf=false; return; end
    end
end
end

function tf=inputsReadyAtPlacement(W,st)
tf=true;
if st.placement_id<=0, tf=false; return; end
for eid=st.pred_edges
    [ed,~]=findEdgeById(W,eid);
    if ed.status~="DELIVERED", tf=false; return; end
    % delivered to the placement vehicle
    if ed.dst_vehicle~=st.placement_id && ~(ed.src_vehicle==ed.dst_vehicle && ed.dst_vehicle==st.placement_id)
        % local deliver sets dst=placement
        if ed.dst_vehicle~=st.placement_id, tf=false; return; end
    end
end
end

function b=inputBytesForStage(W,st)
b=0;
for eid=st.pred_edges
    [ed,~]=findEdgeById(W,eid);
    b=b+ed.size_bytes;
end
end

function W=refreshEligibility(W,wfId)
for i=1:numel(W.stages)
    if W.stages(i).workflow_id~=wfId, continue; end
    st=W.stages(i);
    if st.status=="BLOCKED" && predsDone(W,st)
        st.status="ELIGIBLE";
        W.stages(i)=st;
    end
end
end

function W=tryEnqueueStage(W,wfId,stageId)
[st,si]=findStage(W,wfId,stageId);
if ~(st.status=="COMMITTED" || st.status=="DATA_WAIT"), return; end
if ~inputsReadyAtPlacement(W,st)
    st.status="DATA_WAIT";
    W.stages(si)=st;
    return
end
if st.job_id>0, return; end % already queued
jid=W.next_job; W.next_job=W.next_job+1;
st.job_id=jid;
st.status="QUEUED";
W.stages(si)=st;
wf=W.workflows(wfIndex(W,wfId));
meta=struct('kind',"workflow",'workflow_id',wfId,'stage_id',string(stageId));
vid=st.placement_id;
W.cpu(vid)=WorkflowCpuQueue('enqueue',W.cpu(vid),jid,"workflow", ...
    st.required_cycles,wf.absolute_deadline_s,W.now_s,meta);
W.log=WorkflowMetrics('log',W.log,W.now_s,"STAGE_ENQUEUE",'workflow_id',wfId, ...
    'stage_id',string(stageId),'vehicle_id',vid);
if ~isempty(W.cpu(vid).running) && W.cpu(vid).running.job_id==jid
    [st,si]=findStage(W,wfId,stageId);
    st.status="RUNNING";
    st.start_s=W.now_s;
    W.stages(si)=st;
    if string(stageId)=="D", W.last_d_start_s=W.now_s; end
    W.log=WorkflowMetrics('log',W.log,W.now_s,"STAGE_START",'workflow_id',wfId, ...
        'stage_id',string(stageId),'vehicle_id',vid);
end
end

function W=onCpuComplete(W,job,vehicleId)
if ~isfield(job,'meta') || ~isfield(job.meta,'kind'), return; end
if job.meta.kind=="background"
    W.log=WorkflowMetrics('log',W.log,job.end_s,"BG_DONE",'vehicle_id',vehicleId, ...
        'detail',sprintf('job=%d',job.job_id));
    return
end
wfId=job.meta.workflow_id; stageId=job.meta.stage_id;
[st,si]=findStage(W,wfId,stageId);
if st.status=="EXPIRED" || st.status=="CANCELLED"
    W.log=WorkflowMetrics('log',W.log,job.end_s,"STAGE_DONE_WASTED", ...
        'workflow_id',wfId,'stage_id',string(stageId),'vehicle_id',vehicleId);
    return
end
st.status="DONE";
st.finish_s=job.end_s;
st.remaining_cycles=0;
W.stages(si)=st;
W.log=WorkflowMetrics('log',W.log,job.end_s,"STAGE_DONE",'workflow_id',wfId, ...
    'stage_id',string(stageId),'vehicle_id',vehicleId);
% Materialize outgoing edges READY_HELD
for eid=st.succ_edges
    [ed,ei]=findEdgeById(W,eid);
    ed.src_vehicle=st.placement_id;
    % dst_vehicle known once dest stage placed; else 0
    if ed.dst_stage==""
        ed.dst_vehicle=W.origin_id;
    else
        [dst,~]=findStage(W,wfId,ed.dst_stage);
        if dst.placement_id>0, ed.dst_vehicle=dst.placement_id; end
    end
    ed.status="READY_HELD";
    ed.held=true;
    [ed,~]=WorkflowMode2Adapter('plan_fragments',ed,W.tb_bytes,job.end_s*1000);
    W.edges(ei)=ed;
end
W=refreshEligibility(W,wfId);
W=autoReleaseReady(W,wfId);
end

function W=autoReleaseReady(W,wfId)
% Local edges: immediate. Remote: release only if dest committed + contact;
% otherwise buffer result edges for later contact.
for ei=1:numel(W.edges)
    ed=W.edges(ei);
    if ed.workflow_id~=wfId || ed.status~="READY_HELD", continue; end
    ed=resolveEdgeVehicles(W,ed);
    W.edges(ei)=ed;
    if ed.src_vehicle<=0 || ed.dst_vehicle<=0, continue; end
    if ed.src_vehicle==ed.dst_vehicle
        W=releaseEdge(W,wfId,ed.name);
        continue
    end
    canRelease=false;
    if ed.dst_stage==""
        canRelease=true; % return to origin once D done
    else
        [dst,~]=findStage(W,wfId,ed.dst_stage);
        canRelease=dst.commitment_acked && dst.placement_id==ed.dst_vehicle;
    end
    if ~canRelease, continue; end
    if contactOk(W,ed.src_vehicle,ed.dst_vehicle)
        W=releaseEdge(W,wfId,ed.name);
    else
        already=false;
        for bi=1:numel(W.buffered_result)
            if W.buffered_result(bi).workflow_id==wfId && ...
                    W.buffered_result(bi).edge_name==ed.name
                already=true; break
            end
        end
        if ~already
            W.buffered_result(end+1)=struct('workflow_id',wfId,'stage_id',ed.src_stage, ...
                'edge_name',ed.name,'helper_id',ed.src_vehicle,'ready_s',W.now_s); %#ok<AGROW>
            W.log=WorkflowMetrics('log',W.log,W.now_s,"RESULT_BUFFERED", ...
                'workflow_id',wfId,'edge_id',ed.edge_id,'detail',string(ed.name));
        end
    end
end
end

function W=onEdgeDelivered(W,wfId,edgeName)
[ed,ei]=findEdge(W,wfId,edgeName);
ed.status="DELIVERED";
if ~isfinite(ed.fully_received_s), ed.fully_received_s=W.now_s; end
W.edges(ei)=ed;
W.log=WorkflowMetrics('log',W.log,W.now_s,"EDGE_DELIVERED",'workflow_id',wfId, ...
    'edge_id',ed.edge_id,'detail',string(edgeName));
if ed.dst_stage~=""
    W=tryEnqueueStage(W,wfId,ed.dst_stage);
else
    % terminal return to origin
    W=checkWorkflowSuccess(W,wfId);
end
end

function W=checkWorkflowSuccess(W,wfId)
wi=wfIndex(W,wfId);
wf=W.workflows(wi);
if wf.status~="ACTIVE", return; end
[st,~]=findStage(W,wfId,"D");
[ed,~]=findEdge(W,wfId,"D_return");
if st.status=="DONE" && ed.status=="DELIVERED"
    t=ed.fully_received_s;
    if t<=wf.absolute_deadline_s+1e-15
        wf.status="SUCCEEDED";
        wf.completion_s=t;
        W.workflows(wi)=wf;
        W.log=WorkflowMetrics('log',W.log,t,"WORKFLOW_SUCCESS",'workflow_id',wfId);
    end
end
end

function W=expireWorkflows(W,t)
for i=1:numel(W.workflows)
    wf=W.workflows(i);
    if wf.status~="ACTIVE", continue; end
    if t+1e-15>=wf.absolute_deadline_s
        % allow success check first (caller order)
        W=checkWorkflowSuccess(W,wf.workflow_id);
        wf=W.workflows(i);
        if wf.status=="SUCCEEDED", continue; end
        wf.status="EXPIRED";
        W.workflows(i)=wf;
        W.log=WorkflowMetrics('log',W.log,t,"WORKFLOW_EXPIRED",'workflow_id',wf.workflow_id);
        for s=1:numel(W.stages)
            if W.stages(s).workflow_id==wf.workflow_id && ...
                    ~(W.stages(s).status=="DONE" || W.stages(s).status=="RUNNING")
                W.stages(s).status="EXPIRED";
            end
        end
    end
end
end

function W=dropWorkflow(W,wfId)
%DROP_WORKFLOW Admission reject — expire workflow immediately (no utility).
[~,wi]=findWorkflow(W,wfId);
if wi<=0, return; end
wf=W.workflows(wi);
if wf.status~="ACTIVE", return; end
wf.status="EXPIRED";
W.workflows(wi)=wf;
W.log=WorkflowMetrics('log',W.log,W.now_s,"WORKFLOW_DROP",'workflow_id',wfId);
for s=1:numel(W.stages)
    if W.stages(s).workflow_id==wfId && W.stages(s).status~="DONE"
        W.stages(s).status="EXPIRED";
    end
end
end

function val=assignWorkflowValue(W,releaseS,wfId)
% Phased / fixed values for prioritized utility (WiLab + abstract).
val=1;
vh=12; vl=1; pEarly=0.05; pLate=0.75; split=0.16;
if isfield(W,'value_high'), vh=double(W.value_high); end
if isfield(W,'value_low'), vl=double(W.value_low); end
if isfield(W,'p_high_early'), pEarly=double(W.p_high_early); end
if isfield(W,'p_high_late'), pLate=double(W.p_high_late); end
if isfield(W,'phase_split_s'), split=double(W.phase_split_s); end
if isfield(W,'phased_values') && logical(W.phased_values)
    p=pEarly; if releaseS>=split, p=pLate; end
    seed=1; if isfield(W,'seed'), seed=double(W.seed); end
    u=stableUniformLocal(seed,"wf-value",wfId,1);
    if u<p, val=vh; else, val=vl; end
elseif isfield(W,'value_plan') && numel(W.value_plan)>=wfId
    val=double(W.value_plan(wfId));
end
end

function [wf,wi]=findWorkflow(W,wfId)
wi=0; wf=struct();
for i=1:numel(W.workflows)
    if W.workflows(i).workflow_id==wfId, wi=i; wf=W.workflows(i); return; end
end
end

function u=stableUniformLocal(seed,tag,a,b)
% Deterministic U(0,1) for reproducible phased values (same spirit as bridge RNG).
h=double(seed)*1000003+double(a)*9176+double(b)*13;
if isstring(tag)||ischar(tag)
    tchars=double(char(string(tag)));
    h=h+sum(tchars(:).*((1:numel(tchars))'));
end
u=mod(h*2654435761,2^31)/(2^31);
u=max(min(u,1-eps),eps);
end

function W=flushBuffered(W)
keep=true(1,numel(W.buffered_result));
for i=1:numel(W.buffered_result)
    br=W.buffered_result(i);
    [ed,~]=findEdge(W,br.workflow_id,br.edge_name);
    ed=resolveEdgeVehicles(W,ed);
    if contactOk(W,ed.src_vehicle,ed.dst_vehicle)
        W=releaseEdge(W,br.workflow_id,br.edge_name);
        keep(i)=false;
    end
end
W.buffered_result=W.buffered_result(keep);
end

function ed=resolveEdgeVehicles(W,ed)
if ed.src_stage~=""
    [st,~]=findStage(W,ed.workflow_id,ed.src_stage);
    if st.placement_id>0, ed.src_vehicle=st.placement_id; end
else
    ed.src_vehicle=W.origin_id;
end
if ed.dst_stage~=""
    [st,~]=findStage(W,ed.workflow_id,ed.dst_stage);
    if st.placement_id>0, ed.dst_vehicle=st.placement_id; end
else
    ed.dst_vehicle=W.origin_id;
end
end

function t=cpuCompletionTime(S)
t=Inf;
if isempty(S.running) || S.available_hz<=0, return; end
t=S.last_accounted_s+S.running.remaining/S.available_hz;
end

function [st,idx]=findStage(W,wfId,stageId)
stageId=string(stageId);
for i=1:numel(W.stages)
    if W.stages(i).workflow_id==wfId && W.stages(i).stage_id==stageId
        st=W.stages(i); idx=i; return
    end
end
error('WorkflowEngine:StageNotFound','%d/%s',wfId,stageId);
end

function [ed,idx]=findEdge(W,wfId,name)
name=string(name);
for i=1:numel(W.edges)
    if W.edges(i).workflow_id==wfId && W.edges(i).name==name
        ed=W.edges(i); idx=i; return
    end
end
error('WorkflowEngine:EdgeNotFound','%d/%s',wfId,name);
end

function [ed,idx]=findEdgeById(W,eid)
for i=1:numel(W.edges)
    if W.edges(i).edge_id==eid, ed=W.edges(i); idx=i; return; end
end
error('WorkflowEngine:EdgeIdNotFound','%d',eid);
end

function i=wfIndex(W,wfId)
for i=1:numel(W.workflows)
    if W.workflows(i).workflow_id==wfId, return; end
end
error('WorkflowEngine:WfNotFound','%d',wfId);
end

function W=injectCtrlRx(W,msg,src,dst)
% Direct injection (tests); still goes through protocol handlers.
W=handleCtrl(W,msg,src,dst);
end

function W=injectFragmentRx(W,wfId,edgeName,fragment,fragments,messageId)
W=handleFragment(W,struct('workflow_id',wfId,'edge_name',string(edgeName), ...
    'fragment',fragment,'fragments',fragments,'message_id',messageId));
end

function W=repeatAck(W,ack)
% Idempotent re-delivery of ACK by nonce
for i=1:numel(W.ack_cache)
    if W.ack_cache(i).nonce==ack.nonce
        key=W.ack_cache(i).stage_key;
        parts=split(string(key),"/");
        [st,si]=findStage(W,str2double(parts(1)),parts(2));
        before=st.status;
        nLedger=numel(W.origin_ledger);
        [st,W.origin_ledger]=WorkflowControlProtocol('origin_handle_ack',st,W.origin_ledger,ack,W.now_s*1000);
        if before=="COMMITTED" || before=="DATA_WAIT" || before=="QUEUED" ...
                || before=="RUNNING" || before=="DONE"
            st.status=before;
            % Roll back duplicate ledger growth from protocol helper
            if numel(W.origin_ledger)>nLedger
                W.origin_ledger=W.origin_ledger(1:nLedger);
            end
        end
        W.stages(si)=st;
        W.log=WorkflowMetrics('log',W.log,W.now_s,"ACK_REPEAT",'workflow_id',st.workflow_id, ...
            'stage_id',st.stage_id,'detail',"idempotent");
        return
    end
end
end

function W=repeatFragment(W,wfId,edgeName,fragment,fragments,messageId)
key=sprintf('%d_%s_%d',wfId,edgeName,fragment);
if isKey(W.frag_rx_seen,key) && W.frag_rx_seen(key)
    W.log=WorkflowMetrics('log',W.log,W.now_s,"FRAGMENT_RX_DUP",'workflow_id',wfId, ...
        'detail',key);
    return
end
W=handleFragment(W,struct('workflow_id',wfId,'edge_name',string(edgeName), ...
    'fragment',fragment,'fragments',fragments,'message_id',messageId));
end

function W=handleCtrl(W,msg,src,dst) %#ok<INUSD>
ct=string(msg.ctrl_type);
if ct=="HELPER_ADV"
    % Persist only a received, timestamped helper advertisement.  Policies
    % must read this cache rather than helper CPU truth.
    h=msg.helper_id;
    adv=struct('helper_id',h,'rx_s',W.now_s,'generated_s',msg.generated_ms/1000, ...
        'ttl_s',msg.ttl_ms/1000,'nominal_cpu_hz',msg.nominal_cpu_hz, ...
        'available_cpu_hz',msg.available_cpu_hz,'queued_cycles',msg.queued_cycles, ...
        'running_remaining_cycles',msg.running_remaining_cycles, ...
        'admitted_pending_cycles',msg.admitted_pending_cycles, ...
        'willingness',logical(msg.willingness),'seq',msg.seq);
    found=false;
    for i=1:numel(W.advertisements)
        if W.advertisements(i).helper_id==h
            if adv.seq>=W.advertisements(i).seq, W.advertisements(i)=adv; end
            found=true; break
        end
    end
    if ~found, W.advertisements(end+1)=adv; end %#ok<AGROW>
    W.log=WorkflowMetrics('log',W.log,W.now_s,"HELPER_ADV_RX",'vehicle_id',dst, ...
        'detail',sprintf('helper=%d seq=%d',h,msg.seq));
elseif ct=="COMMIT_REQ"
    helper=msg.helper_id;
    % Map helper_id if message carries WiLab id — callers should use engine indices.
    [W.ledger{helper},resp]=WorkflowControlProtocol('helper_handle_req', ...
        W.ledger{helper},msg,W.willingness(helper),W.max_admitted_cycles,W.now_s*1000);
    W.log=WorkflowMetrics('log',W.log,W.now_s,"FRAGMENT_RX",'vehicle_id',helper, ...
        'detail',"ctrl COMMIT_REQ");
    if contactOk(W,helper,msg.origin_id)
        [W.pending_tx,~]=WorkflowControlProtocol('enqueue_ctrl',W.pending_tx,helper,resp,W.tb_bytes);
        if ~(isfield(W,'mode') && W.mode=="wilab_mode2")
            W=scheduleEvent(W,W.now_s+W.ctrl_delay_s,"CTRL_DELIVER",struct( ...
                'msg',resp,'src',helper,'dst',msg.origin_id));
        end
    end
elseif ct=="COMMIT_ACK"
    % Find stage by nonce
    found=false;
    for i=1:numel(W.stages)
        if W.stages(i).req_nonce==msg.nonce
            st=W.stages(i);
            [st,W.origin_ledger]=WorkflowControlProtocol('origin_handle_ack',st,W.origin_ledger,msg,W.now_s*1000);
            W.stages(i)=st;
            W.ack_cache(end+1)=struct('nonce',msg.nonce,'ack',msg, ...
                'stage_key',sprintf('%d/%s',st.workflow_id,st.stage_id)); %#ok<AGROW>
            W.log=WorkflowMetrics('log',W.log,W.now_s,"COMMIT_ACK",'workflow_id',st.workflow_id, ...
                'stage_id',st.stage_id,'vehicle_id',st.placement_id);
            W=autoReleaseReady(W,st.workflow_id);
            W=tryEnqueueStage(W,st.workflow_id,st.stage_id);
            found=true; break
        end
    end
    if ~found
        W.log=WorkflowMetrics('log',W.log,W.now_s,"COMMIT_ACK_ORPHAN",'detail',sprintf('nonce=%d',msg.nonce));
    end
elseif ct=="COMMIT_NACK"
    for i=1:numel(W.stages)
        if W.stages(i).req_nonce==msg.nonce
            [W.stages(i),W.origin_ledger]=WorkflowControlProtocol('origin_handle_nack', ...
                W.stages(i),W.origin_ledger,msg);
            break
        end
    end
end
end

function W=handleFragment(W,p)
key=sprintf('%d_%s_%d',p.workflow_id,p.edge_name,p.fragment);
if isKey(W.frag_rx_seen,key) && W.frag_rx_seen(key)
    W.log=WorkflowMetrics('log',W.log,W.now_s,"FRAGMENT_RX_DUP",'workflow_id',p.workflow_id, ...
        'detail',key);
    return
end
W.frag_rx_seen(key)=true;
[ed,ei]=findEdge(W,p.workflow_id,p.edge_name);
ed.rx_count=ed.rx_count+1;
ed.status="IN_FLIGHT";
W.log=WorkflowMetrics('log',W.log,W.now_s,"FRAGMENT_RX",'workflow_id',p.workflow_id, ...
    'edge_id',ed.edge_id,'detail',sprintf('data f=%d/%d',p.fragment,p.fragments));
W.edges(ei)=ed;
if ed.rx_count>=p.fragments
    ed.fully_received_s=W.now_s;
    W.edges(ei)=ed;
    % Destination stage stays DATA_WAIT until all inputs reassembled
    if ed.dst_stage~=""
        [st,si]=findStage(W,p.workflow_id,ed.dst_stage);
        if st.status=="COMMITTED" || st.status=="DATA_WAIT" || st.status=="ADMISSION_PENDING"
            st.status="DATA_WAIT";
            W.stages(si)=st;
        end
    end
    W=onEdgeDelivered(W,p.workflow_id,p.edge_name);
end
end

function v=getf(s,f,d)
if isfield(s,f), v=s.(f); else, v=d; end
end
