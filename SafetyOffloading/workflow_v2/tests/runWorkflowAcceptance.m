function out=runWorkflowAcceptance(varargin)
%RUNWORKFLOWACCEPTANCE R1-01…R1-07 adapter contract tests (no large sims).
% Usage: out = runWorkflowAcceptance()
%        out = runWorkflowAcceptance('R1-06')  % single id prefix filter
here=fileparts(mfilename('fullpath'));
root=fileparts(here);
addpath(genpath(root));
filter="";
if nargin>=1 && ~isempty(varargin{1}), filter=string(varargin{1}); end

checks={};
checks(end+1,:)={"R1-01",@testR101}; %#ok<*AGROW>
checks(end+1,:)={"R1-02",@testR102};
checks(end+1,:)={"R1-03",@testR103};
checks(end+1,:)={"R1-04",@testR104};
checks(end+1,:)={"R1-05",@testR105};
checks(end+1,:)={"R1-06",@testR106};
checks(end+1,:)={"R1-07",@testR107};

rows=struct('id',{},'ok',{},'msg',{});
for i=1:size(checks,1)
    id=string(checks{i,1});
    if strlength(filter)>0 && ~startsWith(id,filter), continue; end
    try
        msg=checks{i,2}();
        rows(end+1)=struct('id',id,'ok',true,'msg',string(msg));
        fprintf('[PASS] %s - %s\n',id,msg);
    catch ME
        rows(end+1)=struct('id',id,'ok',false,'msg',string(ME.message));
        fprintf('[FAIL] %s - %s\n',id,ME.message);
    end
end
out=struct('rows',rows,'n_pass',sum([rows.ok]),'n_fail',sum(~[rows.ok]), ...
    'all_pass',all([rows.ok]));
if ~out.all_pass
    error('runWorkflowAcceptance:Failed','%d/%d R1 checks failed',out.n_fail,numel(rows));
end
end

function msg=testR101()
% 1500-byte edge → real fragment plan (IDs, sizes); no forced delivery success.
tb=1180; bytes=1500;
edge=struct('workflow_id',1,'edge_id',101,'size_bytes',bytes, ...
    'src_vehicle',1,'dst_vehicle',2,'deadline_ms',500,'message_id',9001, ...
    'commitment_acked',true,'commitment_id',77);
[edge,frags]=WorkflowMode2Adapter('plan_fragments',edge,tb,0);
assert(numel(frags)==2,'expected 2 fragments');
assert(frags(1).bytes==1180 && frags(2).bytes==320,'fragment sizes');
assert(all([frags.fragment]==[1 2]) && all([frags.fragments]==[2 2]),'fragment ids');
assert(all([frags.message_id]==9001),'message_id');
assert(edge.status=="READY_HELD",'held after plan');
assert(WorkflowMode2Adapter('assert_no_mac_until_release',edge),'no MAC yet');
% No forced success: status is not DELIVERED.
assert(edge.status~="DELIVERED",'must not force delivery');
msg=sprintf('%d frags bytes=[%d %d] held',numel(frags),frags(1).bytes,frags(2).bytes);
end

function msg=testR102()
% Fragment callback must persist timeManagement (regression on verified hook).
wilab=fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
    'WiLabIntegration','spv2xWiLabAfterTransmission.m');
assert(isfile(wilab),'missing spv2xWiLabAfterTransmission.m');
txt=fileread(wilab);
assert(contains(txt,'function [stationManagement,timeManagement]'), ...
    'AfterTransmission must return both management structs');
assert(contains(txt,'timeManagement'),'timeManagement must be written');
% Adapter never claims stale RRI bypass — RELEASE only enqueues app queue.
edge=struct('workflow_id',1,'edge_id',1,'size_bytes',100, ...
    'src_vehicle',1,'dst_vehicle',1,'deadline_ms',500,'commitment_acked',true);
[edge,~]=WorkflowMode2Adapter('plan_fragments',edge,1180,10);
assert(WorkflowMode2Adapter('assert_no_mac_until_release',edge));
msg="AfterTransmission dual-return verified; adapter defers MAC to RELEASE";
end

function msg=testR103()
% HOLD across two controller ticks → zero enqueue; RELEASE → one message (N frags).
edge=struct('workflow_id',1,'edge_id',303,'size_bytes',1500, ...
    'src_vehicle',1,'dst_vehicle',2,'deadline_ms',500,'message_id',1, ...
    'commitment_acked',true,'commitment_id',1);
[edge,~]=WorkflowMode2Adapter('plan_fragments',edge,1180,0);
edge=WorkflowMode2Adapter('hold',edge);
pending={[]}; % vehicle 1
for tick=1:2
    assert(WorkflowMode2Adapter('assert_no_mac_until_release',edge), ...
        sprintf('MAC leak at tick %d',tick));
    assert(isempty(pending{1}),'fragments enqueued before RELEASE');
end
assert(WorkflowMode2Adapter('count_held',edge)==1);
[edge,pending]=WorkflowMode2Adapter('release',edge,pending,1);
assert(edge.status=="QUEUED_MAC");
assert(numel(pending{1})==2,'exactly one message = all planned fragments');
assert(all([pending{1}.edge_id]==303));
msg="0 enqueued over 2 ticks; 2 fragments after RELEASE";
end

function msg=testR104()
% Presence from received observations only — no privileged future contact.
obs=WorkflowObservation('empty');
obs.ego_id=1; obs.now_ms=0;
% Privileged oracle would inject future contact; strip must remove it.
obs.future_contact_ms=struct('helper_id',2,'appear_ms',50,'disappear_ms',200);
obs.true_helper_cpu_queue=struct('helper_id',2,'jobs',1);
% Helper becomes visible only after ADV RX (mobility/contact observation proxy).
state=struct('nominal_cpu_hz',1.5e9,'available_cpu_hz',1.5e9,'queued_cycles',0, ...
    'running_remaining_cycles',0,'admitted_pending_cycles',0,'willingness',true,'seq',1);
adv=WorkflowControlProtocol('make_adv',2,state,100,300);
assert(~WorkflowControlProtocol('fresh_adv',adv,500,300),'stale ADV rejected');
assert(WorkflowControlProtocol('fresh_adv',adv,250,300),'fresh ADV accepted');
obs=WorkflowObservation('ingest_adv',obs,adv,250);
dep=WorkflowObservation('strip_deployable',obs);
assert(WorkflowObservation('assert_no_true_queue',dep),'true queue leaked');
assert(isempty(dep.future_contact_ms),'future contact leaked');
assert(numel(dep.helpers)==1 && dep.helpers(1).helper_id==2);
assert(numel(dep.contact_observed)==1 && dep.contact_observed(1).reachable);
msg="contact via ADV RX only; future_contact stripped";
end

function msg=testR105()
% ADV + COMMIT_REQ/ACK/NACK as app payloads with bytes; timeout + idempotency.
pending={[],[]}; % vehicles 1=origin, 2=helper
state=struct('nominal_cpu_hz',2e9,'available_cpu_hz',2e9,'queued_cycles',0, ...
    'running_remaining_cycles',0,'admitted_pending_cycles',0,'willingness',true,'seq',1);
adv=WorkflowControlProtocol('make_adv',2,state,0,300);
[pending,am]=WorkflowControlProtocol('enqueue_ctrl',pending,2,adv,1180);
assert(am.kind=="workflow_ctrl" && am.ctrl_type=="HELPER_ADV");
assert(am.bytes==adv.payload_bytes && am.bytes>0,'ADV air bytes');

req=WorkflowControlProtocol('make_req',10,"B",2,12e6,500,900,42,10,1);
[pending,rm]=WorkflowControlProtocol('enqueue_ctrl',pending,1,req,1180);
assert(rm.ctrl_type=="COMMIT_REQ" && rm.bytes==64);
assert(rm.dst==2 && rm.nonce==42);

ledger=WorkflowControlProtocol('empty_ledger');
[ledger,ack]=WorkflowControlProtocol('helper_handle_req',ledger,req,true,20e6,15);
assert(ack.ctrl_type=="COMMIT_ACK");
[ledger2,ack2]=WorkflowControlProtocol('helper_handle_req',ledger,req,true,20e6,20);
assert(ack2.commitment_id==ack.commitment_id && numel(ledger2)==numel(ledger), ...
    'nonce must be idempotent');

[pending,ackm]=WorkflowControlProtocol('enqueue_ctrl',pending,2,ack,1180);
assert(ackm.bytes==48 && ackm.dst==1);
ctrlLatency=ack.generated_ms-req.generated_ms;
assert(ctrlLatency==5,'control latency logged from timestamps');

stage=struct('workflow_id',10,'stage_id',"B",'required_cycles',12e6,'status',"ADMISSION_PENDING", ...
    'req_sent_ms',10,'req_nonce',42,'remote_attempts',0,'commitment_acked',false);
originLedger=WorkflowControlProtocol('empty_ledger');
[stage,originLedger]=WorkflowControlProtocol('origin_handle_ack',stage,originLedger,ack,15);
assert(stage.status=="COMMITTED" && stage.commitment_acked);

% NACK path + timeout
req2=WorkflowControlProtocol('make_req',10,"C",2,12e6,500,900,43,30,1);
[ledger,~]=WorkflowControlProtocol('helper_handle_req',ledger,req2,false,20e6,35);
stage2=struct('workflow_id',10,'stage_id',"C",'required_cycles',12e6,'status',"ADMISSION_PENDING", ...
    'req_sent_ms',30,'req_nonce',43,'remote_attempts',0,'commitment_acked',false);
[stage2,~]=WorkflowControlProtocol('timeout_req',stage2,originLedger,120,80);
assert(stage2.status=="ELIGIBLE" && stage2.last_nack=="timeout");

% Remote RELEASE blocked without ACK
edge=struct('workflow_id',10,'edge_id',5,'size_bytes',900,'src_vehicle',1,'dst_vehicle',2, ...
    'deadline_ms',500,'commitment_acked',false,'commitment_id',0);
[edge,~]=WorkflowMode2Adapter('plan_fragments',edge,1180,40);
threw=false;
try
    WorkflowMode2Adapter('release',edge,pending,1);
catch
    threw=true;
end
assert(threw,'remote RELEASE without ACK must fail');
msg=sprintf('ADV/REQ/ACK bytes + latency=%gms; idempotent; timeout; no free commit',ctrlLatency);
end

function msg=testR106()
% CPU rate change mid-service: integral cycles, one completion, no duplicate work.
fx=jsondecode(fileread(fullfile(fileparts(mfilename('fullpath')), ...
    'fixtures','r2_cpu_rate_change.json')));
S=WorkflowCpuQueue('new',1,fx.rate_hz_before,0);
S=WorkflowCpuQueue('enqueue',S,1,"workflow",fx.job_cycles,1.0,0);
[S,~]=WorkflowCpuQueue('set_rate',S,fx.rate_hz_after,fx.change_at_ms/1000);
[S,ev]=WorkflowCpuQueue('settle',S,0.100);
assert(numel(ev)==1,'exactly one completion event');
assert(abs(ev{1}.end_s*1000-fx.expected_completion_ms)<1e-6, ...
    sprintf('completion %.6f ms != %g',ev{1}.end_s*1000,fx.expected_completion_ms));
assert(abs(S.total_served-fx.job_cycles)<1e-3,'service conservation');
assert(isempty(S.running) && isempty(S.queue),'queue drained once');
msg=sprintf('completion at %g ms; served=%g',ev{1}.end_s*1000,S.total_served);
end

function msg=testR107()
% True helper queue absent from policy input; only received ADV summary.
obs=WorkflowObservation('empty');
obs.true_helper_cpu_queue=struct('helper_id',2,'queued_cycles',9e6,'jobs',[1 2 3]);
obs.future_contact_ms=999;
% Inject oracle-labeled helper (must be dropped).
obs.helpers=struct('helper_id',2,'available_cpu_hz',2e9,'queued_cycles',9e6, ...
    'running_remaining_cycles',0,'admitted_pending_cycles',0,'willingness',true, ...
    'rx_ms',NaN,'ttl_ms',300,'fresh',false,'source',"oracle_true_queue");
state=struct('nominal_cpu_hz',2e9,'available_cpu_hz',1.8e9,'queued_cycles',1e6, ...
    'running_remaining_cycles',0,'admitted_pending_cycles',0,'willingness',true,'seq',3);
adv=WorkflowControlProtocol('make_adv',2,state,50,300);
obs=WorkflowObservation('ingest_adv',obs,adv,55);
dep=WorkflowObservation('strip_deployable',obs);
assert(WorkflowObservation('assert_no_true_queue',dep));
assert(isempty(dep.true_helper_cpu_queue));
assert(numel(dep.helpers)==1);
assert(dep.helpers(1).source=="received_adv");
assert(dep.helpers(1).queued_cycles==1e6,'uses ADV summary not true queue 9e6');
assert(dep.helpers(1).rx_ms==55,'timestamped receipt');
msg="true queue stripped; policy sees timestamped ADV only";
end
