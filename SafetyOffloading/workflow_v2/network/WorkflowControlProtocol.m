function varargout=WorkflowControlProtocol(op,varargin)
%WORKFLOWCONTROLPROTOCOL Helper ADV / COMMIT_REQ / ACK / NACK as app payloads.
% Zero-latency free oracle is forbidden. Messages are ordinary Mode-2 payloads
% with bytes and optional enqueue onto pending_tx (same path as data edges).
%
% ops:
%   msg = ...('make_adv', helperId, state, nowMs, ttlMs)
%   msg = ...('make_req', workflowId, stageId, helperId, cycles, deadlineMs, inputBytes, nonce, nowMs)
%   msg = ...('make_ack', commitmentId, expiryMs, nonce, helperId, originId, nowMs)
%   msg = ...('make_nack', reason, nonce, helperId, originId, nowMs)
%   [ledger,resp] = ...('helper_handle_req', ledger, req, willingness, maxCycles, nowMs)
%   [stage,ledger] = ...('origin_handle_ack', stage, ledger, ack, nowMs)
%   [stage,ledger] = ...('origin_handle_nack', stage, ledger, nack)
%   [stage,ledger] = ...('timeout_req', stage, ledger, nowMs, timeoutMs)
%   tf = ...('fresh_adv', adv, nowMs, ttlMs)
%   [pending,meta] = ...('enqueue_ctrl', pendingTx, src, msg, tbBytes)
switch string(op)
    case "make_adv", varargout{1}=makeAdv(varargin{:});
    case "make_req", varargout{1}=makeReq(varargin{:});
    case "make_ack", varargout{1}=makeAck(varargin{:});
    case "make_nack", varargout{1}=makeNack(varargin{:});
    case "helper_handle_req", [varargout{1},varargout{2}]=helperHandleReq(varargin{:});
    case "origin_handle_ack", [varargout{1},varargout{2}]=originHandleAck(varargin{:});
    case "origin_handle_nack", [varargout{1},varargout{2}]=originHandleNack(varargin{:});
    case "timeout_req", [varargout{1},varargout{2}]=timeoutReq(varargin{:});
    case "fresh_adv", varargout{1}=freshAdv(varargin{:});
    case "enqueue_ctrl", [varargout{1},varargout{2}]=enqueueCtrl(varargin{:});
    case "empty_ledger", varargout{1}=emptyLedger();
    otherwise, error('WorkflowControlProtocol:UnknownOp','Unknown op %s',op);
end
end

function L=emptyLedger()
L=struct('nonce',{},'commitment_id',{},'workflow_id',{},'stage_id',{}, ...
    'cycles',{},'status',{},'request_s',{},'ack_s',{},'expire_s',{},'response',{});
end

function msg=makeAdv(helperId,state,nowMs,ttlMs)
if nargin<4, ttlMs=300; end
msg=struct('ctrl_type',"HELPER_ADV",'helper_id',helperId,'generated_ms',nowMs, ...
    'ttl_ms',ttlMs,'nominal_cpu_hz',state.nominal_cpu_hz, ...
    'available_cpu_hz',state.available_cpu_hz,'queued_cycles',state.queued_cycles, ...
    'running_remaining_cycles',state.running_remaining_cycles, ...
    'admitted_pending_cycles',state.admitted_pending_cycles, ...
    'willingness',logical(state.willingness),'seq',state.seq, ...
    'payload_bytes',estimateAdvBytes(),'nonce',0,'commitment_id',0);
end

function msg=makeReq(workflowId,stageId,helperId,cycles,deadlineMs,inputBytes,nonce,nowMs,originId)
if nargin<9, originId=0; end
msg=struct('ctrl_type',"COMMIT_REQ",'workflow_id',workflowId,'stage_id',stageId, ...
    'helper_id',helperId,'origin_id',originId,'cycles',cycles,'deadline_ms',deadlineMs, ...
    'input_bytes',inputBytes,'nonce',nonce,'generated_ms',nowMs, ...
    'payload_bytes',64,'commitment_id',0,'willingness',false,'seq',0, ...
    'nominal_cpu_hz',0,'available_cpu_hz',0,'queued_cycles',0, ...
    'running_remaining_cycles',0,'admitted_pending_cycles',0,'ttl_ms',0);
end

function msg=makeAck(commitmentId,expiryMs,nonce,helperId,originId,nowMs)
msg=struct('ctrl_type',"COMMIT_ACK",'commitment_id',commitmentId,'expire_ms',expiryMs, ...
    'nonce',nonce,'helper_id',helperId,'origin_id',originId,'generated_ms',nowMs, ...
    'payload_bytes',48,'workflow_id',0,'stage_id',0,'cycles',0,'deadline_ms',0, ...
    'input_bytes',0,'willingness',true,'seq',0,'nominal_cpu_hz',0, ...
    'available_cpu_hz',0,'queued_cycles',0,'running_remaining_cycles',0, ...
    'admitted_pending_cycles',0,'ttl_ms',0,'reason',"");
end

function msg=makeNack(reason,nonce,helperId,originId,nowMs)
msg=struct('ctrl_type',"COMMIT_NACK",'reason',string(reason),'nonce',nonce, ...
    'helper_id',helperId,'origin_id',originId,'generated_ms',nowMs, ...
    'payload_bytes',40,'commitment_id',0,'expire_ms',0,'workflow_id',0, ...
    'stage_id',0,'cycles',0,'deadline_ms',0,'input_bytes',0,'willingness',false, ...
    'seq',0,'nominal_cpu_hz',0,'available_cpu_hz',0,'queued_cycles',0, ...
    'running_remaining_cycles',0,'admitted_pending_cycles',0,'ttl_ms',0);
end

function [ledger,resp]=helperHandleReq(ledger,req,willingness,maxCycles,nowMs)
% Idempotent: same nonce returns prior response without double reservation.
for i=1:numel(ledger)
    if ledger(i).nonce==req.nonce
        resp=ledger(i).response;
        return
    end
end
outstanding=0;
for i=1:numel(ledger)
    if ledger(i).status=="RESERVED" || ledger(i).status=="ACKED"
        outstanding=outstanding+ledger(i).cycles;
    end
end
ok=logical(willingness) && (outstanding+req.cycles)<=maxCycles;
entry=struct('nonce',req.nonce,'commitment_id',0,'workflow_id',req.workflow_id, ...
    'stage_id',req.stage_id,'cycles',req.cycles,'status',"","request_s",nowMs/1000, ...
    'ack_s',NaN,'expire_s',NaN,'response',[]);
if ok
    cid=1e6+numel(ledger)+1;
    entry.commitment_id=cid;
    entry.status="ACKED";
    entry.ack_s=nowMs/1000;
    entry.expire_s=req.deadline_ms/1000;
    resp=makeAck(cid,req.deadline_ms,req.nonce,req.helper_id,req.origin_id,nowMs);
else
    entry.status="NACKED";
    resp=makeNack("capacity_or_willingness",req.nonce,req.helper_id,req.origin_id,nowMs);
end
entry.response=resp;
ledger(end+1)=entry; %#ok<AGROW>
end

function [stage,ledger]=originHandleAck(stage,ledger,ack,nowMs)
assert(ack.ctrl_type=="COMMIT_ACK",'expected ACK');
if isfield(stage,'status') && (stage.status=="EXPIRED" || stage.status=="CANCELLED")
    stage.late_ack=true;
    return
end
stage.status="COMMITTED";
stage.commitment_id=ack.commitment_id;
stage.placement_id=ack.helper_id;
stage.commit_ack_ms=nowMs;
stage.commitment_acked=true;
% Mirror ledger on origin for bookkeeping
ledger(end+1)=struct('nonce',ack.nonce,'commitment_id',ack.commitment_id, ...
    'workflow_id',stage.workflow_id,'stage_id',stage.stage_id,'cycles',stage.required_cycles, ...
    'status',"ACKED",'request_s',NaN,'ack_s',nowMs/1000,'expire_s',ack.expire_ms/1000, ...
    'response',ack); %#ok<AGROW>
end

function [stage,ledger]=originHandleNack(stage,ledger,nack)
stage.status="ELIGIBLE";
stage.last_nack=string(nack.reason);
stage.commitment_acked=false;
stage.remote_attempts=stage.remote_attempts+1;
ledger(end+1)=struct('nonce',nack.nonce,'commitment_id',0,'workflow_id',stage.workflow_id, ...
    'stage_id',stage.stage_id,'cycles',0,'status',"NACKED",'request_s',NaN, ...
    'ack_s',NaN,'expire_s',NaN,'response',nack); %#ok<AGROW>
end

function [stage,ledger]=timeoutReq(stage,ledger,nowMs,timeoutMs)
if ~(isfield(stage,'status') && stage.status=="ADMISSION_PENDING"), return; end
if ~isfield(stage,'req_sent_ms'), return; end
if nowMs-stage.req_sent_ms<timeoutMs, return; end
stage.status="ELIGIBLE";
stage.last_nack="timeout";
stage.commitment_acked=false;
stage.remote_attempts=stage.remote_attempts+1;
ledger(end+1)=struct('nonce',stage.req_nonce,'commitment_id',0,'workflow_id',stage.workflow_id, ...
    'stage_id',stage.stage_id,'cycles',0,'status',"TIMEOUT",'request_s',stage.req_sent_ms/1000, ...
    'ack_s',NaN,'expire_s',NaN,'response',[]); %#ok<AGROW>
end

function tf=freshAdv(adv,nowMs,ttlMs)
if nargin<3, ttlMs=adv.ttl_ms; end
tf=(nowMs-adv.generated_ms)<=ttlMs;
end

function [pendingTx,meta]=enqueueCtrl(pendingTx,src,msg,tbBytes)
if nargin<4, tbBytes=1180; end
m=WorkflowMode2Adapter('empty_meta');
m.valid=true; m.kind="workflow_ctrl"; m.ctrl_type=msg.ctrl_type;
m.src=src;
if isfield(msg,'helper_id') && msg.ctrl_type=="COMMIT_REQ"
    m.dst=msg.helper_id;
elseif isfield(msg,'origin_id') && (msg.ctrl_type=="COMMIT_ACK" || msg.ctrl_type=="COMMIT_NACK")
    m.dst=msg.origin_id;
else
    m.dst=0; % broadcast ADV
end
m.bytes=min(tbBytes,max(1,msg.payload_bytes));
m.fragment=1; m.fragments=1; m.seq=1;
m.created_ms=msg.generated_ms; m.nonce=msg.nonce;
m.commitment_id=msg.commitment_id;
if isfield(msg,'workflow_id'), m.workflow_id=msg.workflow_id; end
if isfield(msg,'stage_id'), m.stage_id=string(msg.stage_id); m.stage=string(msg.stage_id); end
% Preserve protocol fields through Mode-2 for RX reconstruction
m.helper_id=0; m.origin_id=0; m.cycles=0; m.input_bytes=0; m.expire_ms=0; m.reason="";
if isfield(msg,'helper_id'), m.helper_id=msg.helper_id; end
if isfield(msg,'origin_id'), m.origin_id=msg.origin_id; end
if isfield(msg,'cycles'), m.cycles=msg.cycles; end
if isfield(msg,'input_bytes'), m.input_bytes=msg.input_bytes; end
if isfield(msg,'expire_ms'), m.expire_ms=msg.expire_ms; end
if isfield(msg,'deadline_ms'), m.deadline_ms=msg.deadline_ms; end
if isfield(msg,'reason'), m.reason=string(msg.reason); end
if isfield(msg,'ttl_ms'), m.ttl_ms=msg.ttl_ms; end
if isfield(msg,'nominal_cpu_hz'), m.nominal_cpu_hz=msg.nominal_cpu_hz; end
if isfield(msg,'available_cpu_hz'), m.available_cpu_hz=msg.available_cpu_hz; end
if isfield(msg,'queued_cycles'), m.queued_cycles=msg.queued_cycles; end
if isfield(msg,'running_remaining_cycles'), m.running_remaining_cycles=msg.running_remaining_cycles; end
if isfield(msg,'admitted_pending_cycles'), m.admitted_pending_cycles=msg.admitted_pending_cycles; end
if isfield(msg,'willingness'), m.willingness=logical(msg.willingness); end
cur=pendingTx{src};
if isempty(cur)
    pendingTx{src}=m;
else
    pendingTx{src}=[cur(:); m];
end
meta=m;
end

function b=estimateAdvBytes()
% Engineering encoding estimate; R1_MEASURE_FROM_ENCODING can refine.
b=80;
end
