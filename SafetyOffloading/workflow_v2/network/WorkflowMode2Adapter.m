function varargout=WorkflowMode2Adapter(op,varargin)
%WORKFLOWMODE2ADAPTER HOLD/RELEASE of whole edges onto Mode-2 app queues.
% Does not select PHY slots. Fragmentation uses configured TB size (measured
% 1180 B assumption) until MAC-effective TB introspection exists.
%
% ops:
%   meta = WorkflowMode2Adapter('empty_meta')
%   [edge,frags] = WorkflowMode2Adapter('plan_fragments',edge,tbBytes,nowMs)
%   edge = WorkflowMode2Adapter('hold',edge)           % READY_HELD, no enqueue
%   [edge,pending] = WorkflowMode2Adapter('release',edge,pendingTxCell,srcVehicle)
%   [edge,pending] = WorkflowMode2Adapter('release_local',edge,nowMs)  % same vehicle
%   n = WorkflowMode2Adapter('count_held',edges)
%   tf = WorkflowMode2Adapter('assert_no_mac_until_release',edge)
switch string(op)
    case "empty_meta"
        varargout{1}=emptyMeta();
    case "plan_fragments"
        [varargout{1},varargout{2}]=planFragments(varargin{:});
    case "hold"
        varargout{1}=holdEdge(varargin{1});
    case "release"
        [varargout{1},varargout{2}]=releaseEdge(varargin{:});
    case "release_local"
        [varargout{1},varargout{2}]=releaseLocal(varargin{:});
    case "count_held"
        varargout{1}=countHeld(varargin{1});
    case "assert_no_mac_until_release"
        varargout{1}=assertNoMac(varargin{1});
    otherwise
        error('WorkflowMode2Adapter:UnknownOp','Unknown op %s',op);
end
end

function m=emptyMeta()
% Includes legacy WiLab fields (task_id/stage) for pending_tx / current(v) compatibility.
m=struct('valid',false,'packet_id',0,'kind',"",'workflow_id',0,'edge_id',0, ...
    'stage_id',"",'fragment',0,'fragments',0,'src',0,'dst',0, ...
    'created_ms',0,'deadline_ms',0,'seq',0,'flow_id',0,'bytes',0, ...
    'message_id',0,'commitment_id',0,'nonce',0,'ctrl_type',"", ...
    'task_id',0,'stage',"",'helper_id',0,'origin_id',0,'cycles',0, ...
    'input_bytes',0,'expire_ms',0,'reason',"");
end

function [edge,frags]=planFragments(edge,tbBytes,nowMs)
if nargin<2 || ~(isfinite(tbBytes) && tbBytes>0), tbBytes=1180; end
if nargin<3, nowMs=0; end
assert(isfield(edge,'size_bytes') && edge.size_bytes>0,'edge size required');
assert(isfield(edge,'src_vehicle') && isfield(edge,'dst_vehicle'),'vehicles required');
n=max(1,ceil(double(edge.size_bytes)/double(tbBytes)));
frags=repmat(emptyMeta(),n,1);
msgId=0;
if isfield(edge,'message_id') && edge.message_id>0, msgId=edge.message_id; end
for f=1:n
    m=emptyMeta(); m.valid=true; m.kind="workflow_data";
    m.workflow_id=edge.workflow_id; m.edge_id=edge.edge_id;
    m.fragment=f; m.fragments=n; m.seq=f;
    m.src=edge.src_vehicle; m.dst=edge.dst_vehicle;
    m.created_ms=nowMs;
    if isfield(edge,'deadline_ms'), m.deadline_ms=edge.deadline_ms; end
    m.bytes=min(tbBytes,edge.size_bytes-(f-1)*tbBytes);
    m.message_id=msgId; m.flow_id=1e9+double(edge.edge_id);
    if isfield(edge,'commitment_id'), m.commitment_id=edge.commitment_id; end
    frags(f)=m;
end
edge.fragment_ids=arrayfun(@(x)x.seq,frags);
edge.planned_fragments=n;
edge.planned_frags=frags;
edge.tb_bytes_assumed=tbBytes;
edge.status="READY_HELD";
edge.release_s=NaN;
edge.mac_enqueued=false;
edge.held=true;
edge.commitment_acked=isfield(edge,'commitment_acked') && edge.commitment_acked;
if edge.src_vehicle==edge.dst_vehicle, edge.commitment_acked=true; end
end

function edge=holdEdge(edge)
% HOLD: materialized but not submitted to MAC.
assert(isfield(edge,'status'),'edge.status required');
if edge.status=="QUEUED_MAC" || edge.status=="IN_FLIGHT" || edge.status=="DELIVERED"
    error('WorkflowMode2Adapter:CannotHold','Cannot HOLD after MAC enqueue');
end
edge.status="READY_HELD";
edge.held=true;
end

function [edge,pendingTx]=releaseEdge(edge,pendingTx,srcVehicle)
% RELEASE: append planned fragments to vehicle application TX queue.
assert(edge.status=="READY_HELD" || edge.status=="WAIT_ADMISSION", ...
    'WorkflowMode2Adapter:BadState','RELEASE requires READY_HELD/WAIT_ADMISSION');
if edge.src_vehicle~=edge.dst_vehicle
    assert(isfield(edge,'commitment_acked') && edge.commitment_acked, ...
        'WorkflowMode2Adapter:NoAck','Remote RELEASE requires commitment ACK');
end
if ~isfield(edge,'planned_frags') || isempty(edge.planned_frags)
    error('WorkflowMode2Adapter:NoPlan','Call plan_fragments before release');
end
frags=edge.planned_frags;
if ~iscell(pendingTx) || numel(pendingTx)<srcVehicle
    error('WorkflowMode2Adapter:PendingShape','pendingTx cell must cover srcVehicle');
end
cur=pendingTx{srcVehicle};
if isempty(cur)
    pendingTx{srcVehicle}=frags(:);
else
    pendingTx{srcVehicle}=[cur(:); frags(:)];
end
edge.status="QUEUED_MAC";
edge.held=false;
edge.release_s=frags(1).created_ms/1000;
edge.mac_enqueued=true;
end

function [edge,pendingTx]=releaseLocal(edge,nowMs)
% Same-vehicle edge: zero radio, immediate DELIVERED.
pendingTx=[]; %#ok<NASGU>
assert(edge.src_vehicle==edge.dst_vehicle,'local release requires same vehicle');
edge.status="DELIVERED";
edge.fully_received_s=nowMs/1000;
edge.held=false;
edge.mac_enqueued=false;
edge.fragment_ids=[];
pendingTx={}; % unused
end

function n=countHeld(edges)
n=0;
for i=1:numel(edges)
    if isfield(edges(i),'status') && edges(i).status=="READY_HELD", n=n+1; end
end
end

function tf=assertNoMac(edge)
tf=~isfield(edge,'mac_enqueued') || ~edge.mac_enqueued;
if isfield(edge,'status')
    tf=tf && edge.status~="QUEUED_MAC" && edge.status~="IN_FLIGHT";
end
end
