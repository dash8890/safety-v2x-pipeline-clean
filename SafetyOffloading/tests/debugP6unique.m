function debugP6unique
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));
s=struct();
s.cfg.radio=struct('max_transport_block_bytes',1180,'selection_window_ms',20,'slot_duration_ms',0.5);
s.cfg.compute=struct('deadline_ms',100,'input_bytes',1500,'output_bytes',300,'cycles_per_task',36e6, ...
    'helper_range_m',300,'max_helpers',4,'arrival_rate_tasks_per_s',6);
s.cfg.control=struct('schedule_interval_ms',5,'risk_harm_limit',0.01);
s.origin_ids=1; s.helper_ids=[2;3];
s.cpu_rate=[0.5e9;2e9;1e9]; s.cpu_queue={[],[],[]}; s.cpu_active=zeros(3,1); s.cpu_finish_s=inf(3,1);
s.pending_tx={[],[],[]}; s.packet_counter=0; s.offload_count=0; s.tentative_offloads=0;
s.empty_meta=struct('valid',false,'packet_id',0,'kind',"",'task_id',0,'stage',"",'fragment',0, ...
    'fragments',0,'src',0,'dst',0,'created_ms',0,'deadline_ms',0,'seq',0,'flow_id',0,'bytes',0);
t=struct('task_id',1,'origin',1,'release_ms',0,'deadline_ms',100,'input_bytes',1500,'output_bytes',300, ...
    'cycles',36e6,'value',1,'action',"pending",'helper',0,'status',"pending_unassigned", ...
    'input_fragments',0,'input_received',0,'output_fragments',0,'output_received',0,'cpu_owner',0, ...
    'cpu_start_ms',NaN,'cpu_finish_ms',NaN,'input_delivered_ms',NaN,'result_delivered_ms',NaN,'decision_id',0);
s.tasks=[t;t;t]; s.tasks(2).task_id=2; s.tasks(3).task_id=3;
stationManagement=struct('cbrCV2Xvalues',zeros(3,1));
positionManagement=struct('distanceReal',[0 40 70; 40 0 50; 70 50 0]);
tentative=spv2xNewTentativeState(s);
obs=spv2xDeployableObservation(spv2xMaskActions( ...
    spv2xBuildObservation(s,[1 2 3],stationManagement,positionManagement,0,tentative),s,tentative));
fprintf('assignments=%d masked=%d\n',numel(obs.assignments),sum([obs.assignments.masked]));
plan=spv2xRiskConstrainedMyopicPlan(obs,s,tentative);
fprintf('plan length=%d\n',numel(plan));
for i=1:numel(plan), fprintf('  %s task=%d proc=%d\n',plan(i).type,plan(i).task_id,plan(i).processor_id); end
[s2,epochLog]=spv2xScheduleEpoch(s,[1 2 3],stationManagement,positionManagement,0,"risk_myopic_p6");
fprintf('actions=%d\n',numel(epochLog.actions));
for i=1:numel(epochLog.actions)
    a=epochLog.actions{i};
    fprintf('  action %s task=%d proc=%d typclass=%s\n',string(a.type),a.task_id,a.processor_id,class(a.type));
end
end
