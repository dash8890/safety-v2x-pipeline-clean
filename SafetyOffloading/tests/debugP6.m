function debugP6
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));
try
    runP6AcceptanceSuite();
catch
end
% Re-run first failing check with stack
s=mock();
stationManagement=struct('cbrCV2Xvalues',zeros(3,1));
positionManagement=struct('distanceReal',[0 40 70; 40 0 50; 70 50 0]);
tentative=spv2xNewTentativeState(s);
obs=spv2xDeployableObservation(spv2xMaskActions( ...
    spv2xBuildObservation(s,[1 2],stationManagement,positionManagement,0,tentative),s,tentative));
try
    plan=spv2xRiskConstrainedMyopicPlan(obs,s,tentative);
    disp('plan ok'); disp(plan);
catch ME
    disp(getReport(ME,'extended'));
end
end

function s=mock()
s=struct();
s.cfg.radio=struct('max_transport_block_bytes',1180,'selection_window_ms',20,'slot_duration_ms',0.5);
s.cfg.compute=struct('deadline_ms',100,'input_bytes',1500,'output_bytes',300,'cycles_per_task',36e6, ...
    'helper_range_m',300,'max_helpers',4,'arrival_rate_tasks_per_s',6);
s.cfg.control=struct('risk_harm_limit',0.01);
s.origin_ids=1; s.helper_ids=[2;3];
s.cpu_rate=[0.5e9;2e9;1.5e9]; s.cpu_queue={[],[],[]}; s.cpu_active=zeros(3,1); s.cpu_finish_s=inf(3,1);
s.pending_tx={[],[],[]}; s.packet_counter=0; s.offload_count=0; s.tentative_offloads=0;
s.empty_meta=struct('valid',false,'packet_id',0,'kind',"",'task_id',0,'stage',"",'fragment',0, ...
    'fragments',0,'src',0,'dst',0,'created_ms',0,'deadline_ms',0,'seq',0,'flow_id',0,'bytes',0);
t=struct('task_id',1,'origin',1,'release_ms',0,'deadline_ms',100,'input_bytes',1500,'output_bytes',300, ...
    'cycles',36e6,'value',1,'action',"pending",'helper',0,'status',"pending_unassigned", ...
    'input_fragments',0,'input_received',0,'output_fragments',0,'output_received',0,'cpu_owner',0, ...
    'cpu_start_ms',NaN,'cpu_finish_ms',NaN,'input_delivered_ms',NaN,'result_delivered_ms',NaN,'decision_id',0);
s.tasks=[t;t]; s.tasks(2).task_id=2;
end
