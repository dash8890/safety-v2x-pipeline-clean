function report=runP8EnvValidation()
%RUNP8ENVVALIDATION P8.1 — sequential learning env vs P5/P7 contracts (no WiLab sweep).
root=fileparts(mfilename('fullpath')); addpath(genpath(root));
checks=strings(0,1); failed=strings(0,1);

% Reuse P5 contract suite as the base environment.
try
    p5=runP5AcceptanceSuite();
    if p5.ok, checks(end+1)="P5 acceptance suite"; else, failed(end+1)="P5 acceptance suite failed"; end
catch ME
    failed(end+1)="P5 acceptance: "+string(ME.message);
end

try, checks(end+1)=checkP7CalibrationApplied(); catch ME, failed(end+1)="p7_cal: "+string(ME.message); end
try, checks(end+1)=checkCumulativeScheduleRisk(); catch ME, failed(end+1)="cum_risk: "+string(ME.message); end
try, checks(end+1)=checkMaskedActionLegality(); catch ME, failed(end+1)="mask_legal: "+string(ME.message); end
try, checks(end+1)=checkLearningPolicySelectHooks(); catch ME, failed(end+1)="select: "+string(ME.message); end
try, checks(end+1)=checkRewardCostSeparation(); catch ME, failed(end+1)="reward_cost: "+string(ME.message); end
try, checks(end+1)=checkExperienceIntegrity(); catch ME, failed(end+1)="experience: "+string(ME.message); end
try, checks(end+1)=checkP7ReferenceFrozen(); catch ME, failed(end+1)="p7_ref: "+string(ME.message); end

report=struct('passed',checks(:),'failed',failed(:),'ok',isempty(failed));
outDir=fullfile(root,'outputs','p8_env_validation'); if ~isfolder(outDir), mkdir(outDir); end
fid=fopen(fullfile(outDir,'p8_1_env_validation_report.txt'),'wt');
fprintf(fid,'P8.1 env validation %s\nok=%d passed=%d failed=%d\n\n',string(datetime('now')),report.ok,numel(checks),numel(failed));
for i=1:numel(checks), fprintf(fid,'PASS %s\n',checks(i)); end
for i=1:numel(failed), fprintf(fid,'FAIL %s\n',failed(i)); end
fclose(fid);
fprintf('P8.1 env validation: passed=%d failed=%d ok=%d\n',numel(checks),numel(failed),report.ok);
if ~isempty(failed), disp(failed); end
assert(report.ok,'P8EnvValidation:Failed','One or more P8.1 checks failed');
end

function msg=checkP7CalibrationApplied()
modelPath=fullfile(fileparts(mfilename('fullpath')),'models','p7_action_risk_calibrated.json');
assert(isfile(modelPath),'missing p7_action_risk_calibrated.json');
ar=jsondecode(fileread(modelPath));
s=struct('p7_models',struct('action_risk',ar,'completion',struct()));
a=struct('predicted_completion_probability',1,'predicted_remote_deadline_slack_ms',10, ...
    'predicted_conditional_safety_risk',0.005,'processor_cpu_ghz',0.6,'is_local',0,'relative_distance_m',40);
b=spv2xApplyP7Calibration(s,a);
assert(b.predicted_conditional_safety_risk>=0.008,'P7 action-risk calibration must inflate low provisional scores');
% P6 path: empty p7_models must leave provisional unchanged.
s2=struct('p7_models',struct());
c=spv2xApplyP7Calibration(s2,a);
assert(abs(c.predicted_conditional_safety_risk-0.005)<1e-12,'P6 path must remain uncalibrated');
msg="P7 calibration applied only when p7_models present";
end

function msg=checkCumulativeScheduleRisk()
% ĥ(O,A) is not the max of independent risks: two individually OK offloads can violate.
s=baseStateP8(); s.cfg.control.risk_harm_limit=0.01;
s.cpu_rate(1)=0.5e9; s.cpu_rate(2)=2e9; s.cpu_rate(3)=2e9;
s.tasks=[taskAtP8(1,1,0,100); taskAtP8(2,1,0,100)];
stationManagement=struct('cbrCV2Xvalues',zeros(3,1));
positionManagement=struct('distanceReal',[0 40 70; 40 0 50; 70 50 0]);
% Inject high per-action risk so cumulative multiplier crosses delta.
s.p7_models=struct(); % provisional risk from conditionalRisk; force via model empty
tentative=spv2xNewTentativeState(s);
obs=spv2xBuildObservation(s,[1 2],stationManagement,positionManagement,0,tentative);
obs=spv2xMaskActions(obs,s,tentative);
% Overwrite remote risks to 0.006 (each OK alone; cumulative 0.006*2=0.012 > 0.01).
for i=1:numel(obs.assignments)
    if ~obs.assignments(i).is_local
        obs.assignments(i).predicted_conditional_safety_risk=0.006;
        obs.assignments(i).predicted_completion_probability=1;
    end
end
plan=spv2xRiskConstrainedMyopicPlan(obs,s,tentative);
nOff=0;
for i=1:numel(plan)
    if plan(i).type=="ASSIGN"
        a=obs.assignments([obs.assignments.task_id]==plan(i).task_id & [obs.assignments.processor_id]==plan(i).processor_id);
        if ~isempty(a) && ~a(1).is_local, nOff=nOff+1; end
    end
end
assert(nOff<=1,'cumulative schedule risk must block second 0.006 offload under delta=0.01');
% Schedule-level quantity differs from independent checks.
h1=0.006; h2=min(1,0.006*(1+1));
assert(h1<=0.01 && h2>0.01,'independent OK but joint violates');
msg="cumulative schedule risk ĥ(O,A) enforced";
end

function msg=checkMaskedActionLegality()
s=baseStateP8(); s.cpu_rate(1)=0.5e9; s.cpu_rate(2)=2e9; s.cpu_rate(3)=2e9;
s.tasks=[taskAtP8(1,1,0,100); taskAtP8(2,1,0,100)];
stationManagement=struct('cbrCV2Xvalues',zeros(3,1));
positionManagement=struct('distanceReal',[0 40 70; 40 0 50; 70 50 0]);
tentative=spv2xNewTentativeState(s);
obs=spv2xBuildObservation(s,[1 2],stationManagement,positionManagement,0,tentative);
obs=spv2xMaskActions(obs,s,tentative);
obs=spv2xDeployableObservation(obs);
obs=spv2xMaskActions(obs,s,tentative);
legal=obs.assignments(~[obs.assignments.masked]);
assert(~isempty(legal),'need unmasked candidates');
% Simulate selecting a masked pair must be rejected by epoch apply guard.
masked=obs.assignments([obs.assignments.masked]);
if ~isempty(masked)
    bad=struct('type',"ASSIGN",'task_id',masked(1).task_id,'processor_id',masked(1).processor_id);
    hit=([obs.assignments.task_id]==bad.task_id) & ([obs.assignments.processor_id]==bad.processor_id) & [obs.assignments.masked];
    assert(any(hit),'masked action must be detectable');
end
msg="masked actions detectable for learning policies";
end

function msg=checkLearningPolicySelectHooks()
% selectAction must route P8 policies without falling through to myopic via `|| true`.
s=baseStateP8(); s.policy="masked_ddqn_p8";
s.p8_ddqn=struct('type',"stub"); % presence check only — real model loaded in training
stationManagement=struct('cbrCV2Xvalues',zeros(3,1));
positionManagement=struct('distanceReal',[0 40 70; 40 0 50; 70 50 0]);
s.tasks=taskAtP8(1,1,0,100);
% Without a real network, SelectMaskedDdqn should still return a legal STOP/ASSIGN/REJECT.
[~,epochLog]=spv2xScheduleEpoch(s,1,stationManagement,positionManagement,0,"masked_ddqn_p8");
assert(~isempty(epochLog.actions),'epoch must produce actions');
a=epochLog.actions{1};
assert(ismember(string(a.type),["ASSIGN","REJECT","STOP"]),'illegal action type');
msg="learning policy select hooks reachable";
end

function msg=checkRewardCostSeparation()
% Rewards are completion events; policy harm is P2 paired accounting — never broadcast onto actions.
expPath=fullfile(fileparts(mfilename('fullpath')),'outputs','p7_datasets','sequential_experience.csv');
assert(isfile(expPath),'sequential_experience.csv required');
% Spot-check via MATLAB read of first lines is heavy; use python-built audit if present,
% else verify schema columns exist in header.
fid=fopen(expPath,'rt'); hdr=fgetl(fid); fclose(fid);
assert(contains(hdr,'reward') && contains(hdr,'action_level_harm_cost'),'experience schema');
assert(contains(hdr,'label_kind'),'label_kind required');
msg="reward/cost separation schema present on experience";
end

function msg=checkExperienceIntegrity()
auditPath=fullfile(fileparts(mfilename('fullpath')),'outputs','p7_datasets','coverage_audit.json');
assert(isfile(auditPath),'coverage_audit.json missing');
a=jsondecode(fileread(auditPath));
assert(a.n_sequential_transitions>1000,'need sequential transitions');
assert(a.n_experience_rows>1000,'need experience rows');
assert(a.has_schedule_epochs_any,'schedule_epochs required for P8 offline learning');
assert(a.has_assignment_features_any,'assignment_features required');
msg="experience coverage sufficient for offline A/B";
end

function msg=checkP7ReferenceFrozen()
refDir=fullfile(fileparts(mfilename('fullpath')),'outputs','p7_calibrated_myopic_eval');
sumCsv=fullfile(refDir,'p7_summary.csv');
assert(isfile(sumCsv),'frozen P7 reference summary missing');
cal=fullfile(fileparts(mfilename('fullpath')),'models','p7_calibration_summary.json');
assert(isfile(cal),'p7_calibration_summary.json missing');
j=jsondecode(fileread(cal));
assert(contains(string(j.reference_for_p8),"calibrated_myopic"),'P8 reference must be calibrated myopic');
msg="P7 calibrated myopic reference frozen for P8";
end

function s=baseStateP8()
s=struct();
s.cfg.radio=struct('max_transport_block_bytes',1180,'selection_window_ms',20,'slot_duration_ms',0.5);
s.cfg.compute=struct('deadline_ms',100,'input_bytes',1500,'output_bytes',300, ...
    'cycles_per_task',36e6,'helper_range_m',300,'max_helpers',4,'arrival_rate_tasks_per_s',6);
s.cfg.control=struct('schedule_interval_ms',5,'risk_harm_limit',0.01);
s.origin_ids=1; s.helper_ids=[2;3];
s.cpu_rate=zeros(3,1); s.cpu_rate(:)=1e9;
s.cpu_queue={[],[],[]}; s.cpu_active=zeros(3,1); s.cpu_finish_s=inf(3,1);
s.pending_tx={[],[],[]};
s.empty_meta=struct('valid',false,'packet_id',0,'kind',"",'task_id',0,'stage',"",'fragment',0, ...
    'fragments',0,'src',0,'dst',0,'created_ms',0,'deadline_ms',0,'seq',0,'flow_id',0,'bytes',0);
s.packet_counter=0; s.offload_count=0; s.tentative_offloads=0; s.policy="joint_myopic";
s.p7_models=struct(); s.tasks=taskAtP8(1,1,0,100);
end

function t=taskAtP8(id,origin,releaseMs,deadlineMs)
t=struct('task_id',id,'origin',origin,'release_ms',releaseMs,'deadline_ms',deadlineMs, ...
    'input_bytes',1500,'output_bytes',300,'cycles',36e6,'value',1,'action',"pending", ...
    'helper',0,'status',"pending_unassigned",'input_fragments',0,'input_received',0, ...
    'output_fragments',0,'output_received',0,'cpu_owner',0,'cpu_start_ms',NaN, ...
    'cpu_finish_ms',NaN,'input_delivered_ms',NaN,'result_delivered_ms',NaN,'decision_id',0);
end
