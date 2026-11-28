function TestP5ObservationInterface
%TESTP5OBSERVATIONINTERFACE Lightweight P5 interface checks (no WiLab).
s=mockState();
tentative=spv2xNewTentativeState(s);
[cls,localSlack,remoteSlack]=spv2xFeasibilityClass(s,1,[2],0,tentative);
assert(ismember(cls,["I","II","III","IV"]),'feasibility class missing');
assert(isfinite(localSlack) && isfinite(remoteSlack),'slacks must be finite');

stationManagement=struct('cbrCV2Xvalues',zeros(3,1));
positionManagement=struct('distanceReal',[0 50 80; 50 0 40; 80 40 0]);
obs=spv2xBuildObservation(s,1,stationManagement,positionManagement,0,tentative);
assert(~isempty(obs.tasks) && ~isempty(obs.assignments),'observation groups empty');
assert(any([obs.assignments.is_local]) && any(~[obs.assignments.is_local]), ...
    'local and remote assignment features required');
obs=spv2xMaskActions(obs,s,tentative);
assert(all(isfield(obs.assignments,{'masked','mask_reason'})),'mask fields missing');

tentative=spv2xApplyAssignment(s,tentative,1,1,0);
assert(ismember(1,tentative.assigned_tasks),'ASSIGN did not commit task');
assert(tentative.extra_cycles(1)>0,'ASSIGN did not update CPU commitment');

[s2,epochLog]=spv2xScheduleEpoch(s,1,stationManagement,positionManagement,0,"joint_myopic");
assert(~isempty(epochLog.actions),'epoch produced no actions');
assert(s2.tasks(1).action=="local" || s2.tasks(1).action=="offload" || s2.tasks(1).action=="reject", ...
    'epoch did not assign/reject');
fprintf('TestP5ObservationInterface passed (class=%s localSlack=%.1f remoteSlack=%.1f action=%s)\n', ...
    cls,localSlack,remoteSlack,epochLog.actions{1}.type);
end

function s=mockState()
s=struct();
s.cfg=struct();
s.cfg.radio=struct('max_transport_block_bytes',1180,'selection_window_ms',20,'slot_duration_ms',0.5);
s.cfg.compute=struct('deadline_ms',100,'input_bytes',1500,'output_bytes',300, ...
    'cycles_per_task',36e6,'helper_range_m',300,'max_helpers',4,'arrival_rate_tasks_per_s',6);
s.cfg.control=struct('schedule_interval_ms',5);
s.origin_ids=1; s.helper_ids=[2;3];
s.cpu_rate=zeros(3,1); s.cpu_rate(1)=0.5e9; s.cpu_rate(2)=2e9; s.cpu_rate(3)=1.5e9;
s.cpu_queue={[],[],[]}; s.cpu_active=zeros(3,1); s.cpu_finish_s=inf(3,1);
s.pending_tx={[],[],[]}; s.empty_meta=struct('valid',false,'packet_id',0,'kind',"",'task_id',0, ...
    'stage',"",'fragment',0,'fragments',0,'src',0,'dst',0,'created_ms',0,'deadline_ms',0, ...
    'seq',0,'flow_id',0,'bytes',0);
s.packet_counter=0; s.offload_count=0; s.tentative_offloads=0;
task=struct('task_id',1,'origin',1,'release_ms',0,'deadline_ms',100,'input_bytes',1500, ...
    'output_bytes',300,'cycles',36e6,'value',1,'action',"pending",'helper',0, ...
    'status',"pending_unassigned",'input_fragments',0,'input_received',0, ...
    'output_fragments',0,'output_received',0,'cpu_owner',0,'cpu_start_ms',NaN, ...
    'cpu_finish_ms',NaN,'input_delivered_ms',NaN,'result_delivered_ms',NaN,'decision_id',0);
s.tasks=task;
end
