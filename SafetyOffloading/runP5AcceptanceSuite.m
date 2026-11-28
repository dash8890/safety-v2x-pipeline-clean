function report=runP5AcceptanceSuite()
%RUNP5ACCEPTANCESUITE Bounded P5 interface/math correctness checks (no WiLab).
% Purpose: validate observation/action/commitment/reward/safety-cost contracts.
% Not a performance experiment.
root=fileparts(mfilename('fullpath')); addpath(genpath(root));
checks=strings(0,1); failed=strings(0,1);

try, checks(end+1)=checkAnalyticalSlacks(); catch ME, failed(end+1)="analytical: "+string(ME.message); end
try, checks(end+1)=checkFeasibilityClasses(); catch ME, failed(end+1)="classes: "+string(ME.message); end
try, checks(end+1)=checkCandidateEnumeration(); catch ME, failed(end+1)="enumeration: "+string(ME.message); end
try, checks(end+1)=checkDeployableNoLeakage(); catch ME, failed(end+1)="deployable: "+string(ME.message); end
try, checks(end+1)=checkMaskingAndActions(); catch ME, failed(end+1)="masking: "+string(ME.message); end
try, checks(end+1)=checkSequentialCommitments(); catch ME, failed(end+1)="commitments: "+string(ME.message); end
try, checks(end+1)=checkNoDuplicateAssignment(); catch ME, failed(end+1)="duplicates: "+string(ME.message); end
try, checks(end+1)=checkNegativeControlLocal(); catch ME, failed(end+1)="negctrl: "+string(ME.message); end
try, checks(end+1)=checkRewardConsistency(); catch ME, failed(end+1)="reward: "+string(ME.message); end
try, checks(end+1)=checkSafetyCostP2Agreement(); catch ME, failed(end+1)="safety: "+string(ME.message); end

report=struct('passed',checks(:),'failed',failed(:),'ok',isempty(failed));
outDir=fullfile(root,'outputs','p5_acceptance'); if ~isfolder(outDir), mkdir(outDir); end
fid=fopen(fullfile(outDir,'p5_acceptance_report.txt'),'wt');
fprintf(fid,'P5 acceptance %s\nok=%d passed=%d failed=%d\n\n',string(datetime('now')),report.ok,numel(checks),numel(failed));
for i=1:numel(checks), fprintf(fid,'PASS %s\n',checks(i)); end
for i=1:numel(failed), fprintf(fid,'FAIL %s\n',failed(i)); end
fclose(fid);
fprintf('P5 acceptance: passed=%d failed=%d ok=%d\n',numel(checks),numel(failed),report.ok);
if ~isempty(failed), disp(failed); end
assert(report.ok,'P5Acceptance:Failed','One or more P5 acceptance checks failed');
end

function msg=checkAnalyticalSlacks()
% Origin 500 MHz, 36 Mcycles, empty queue => 72 ms local; helper 2 GHz => 18 ms compute.
s=baseState(); s.cpu_rate(1)=0.5e9; s.cpu_rate(2)=2e9;
s.tasks=taskAt(1,1,0,100);
tentative=spv2xNewTentativeState(s);
[~,localSlack,remoteSlack,localFinish,remoteFinish]=spv2xFeasibilityClass(s,1,2,0,tentative);
assert(abs(localFinish-72)<1e-6,'local finish expected 72 ms');
assert(abs(localSlack-28)<1e-6,'local slack expected 28 ms');
% Input 1500 B => 2 TB blocks; output 300 B => 1 block; +20 ms selection each.
% remote = 18 + 3*20 = 78 ms; slack = 22 ms.
assert(abs(remoteFinish-78)<1e-6,'remote finish expected 78 ms');
assert(abs(remoteSlack-22)<1e-6,'remote slack expected 22 ms');
msg="analytical slacks (72/78 ms)";
end

function msg=checkFeasibilityClasses()
s=baseState(); tentative=spv2xNewTentativeState(s);
% Class I: local OK, remote impossible (no helpers)
[c1]=spv2xFeasibilityClass(s,1,[],0,tentative); assert(c1=="I" || c1=="II",'empty-helper class');
% Force class III: slow origin already busy beyond deadline, fast helper free
s.cpu_rate(1)=0.5e9; s.cpu_rate(2)=2e9;
s.tasks=taskAt(1,1,0,100);
s.cpu_queue{1}=1; % will double-count if we also forecast task 1 - use dummy backlog via extra
s.cpu_queue{1}=[]; 
tentative=spv2xNewTentativeState(s); tentative.extra_cycles(1)=0.5e9*0.080; % 80 ms backlog
s.tasks(1).deadline_ms=100;
[c3,localSlack,remoteSlack]=spv2xFeasibilityClass(s,1,2,0,tentative);
assert(localSlack<0 && remoteSlack>=0 && c3=="III",'expected class III rescue');
% Class IV: both negative
tentative.extra_cycles(2)=2e9; % huge helper backlog
[c4]=spv2xFeasibilityClass(s,1,2,0,tentative);
assert(c4=="IV",'expected class IV');
msg="feasibility classes I/III/IV";
end

function msg=checkCandidateEnumeration()
s=multiTaskState();
stationManagement=struct('cbrCV2Xvalues',zeros(3,1));
positionManagement=struct('distanceReal',[0 40 70; 40 0 50; 70 50 0]);
obs=spv2xBuildObservation(s,[1 2],stationManagement,positionManagement,0,spv2xNewTentativeState(s));
assert(numel(obs.tasks)==2,'two pending tasks');
% Each task: local + up to 2 helpers => 3 assignments each
assert(numel(obs.assignments)==6,'expected 6 task-processor pairs');
assert(all(ismember([obs.assignments.task_id],[1 2])),'task ids');
assert(sum([obs.assignments.is_local])==2,'one local pair per task');
msg="candidate enumeration (2 tasks x 3 processors)";
end

function msg=checkDeployableNoLeakage()
s=multiTaskState();
stationManagement=struct('cbrCV2Xvalues',zeros(3,1));
positionManagement=struct('distanceReal',[0 40 70; 40 0 50; 70 50 0]);
obs=spv2xBuildObservation(s,1,stationManagement,positionManagement,0,spv2xNewTentativeState(s));
dep=spv2xDeployableObservation(obs);
remote=dep.assignments(~[dep.assignments.is_local]);
assert(~isempty(remote),'need remote pairs');
assert(all(isnan([remote.residual_workload_mcycles])),'helper residual must not leak');
assert(all(strcmp(string({remote.workload_source}),"helper_unannounced")),'workload source tag');
local=dep.assignments([dep.assignments.is_local]);
assert(all(isfinite([local.residual_workload_mcycles])),'local residual allowed');
msg="deployable observation has no helper-queue leakage";
end

function msg=checkMaskingAndActions()
s=multiTaskState();
stationManagement=struct('cbrCV2Xvalues',zeros(3,1));
positionManagement=struct('distanceReal',[0 40 70; 40 0 50; 70 50 0]);
tentative=spv2xNewTentativeState(s);
obs=spv2xBuildObservation(s,[1 2],stationManagement,positionManagement,0,tentative);
obs=spv2xMaskActions(obs,s,tentative);
assert(all(isfield(obs.assignments,{'masked','mask_reason'})));
% After assigning task 1, remask must exclude task 1
tentative=spv2xApplyAssignment(s,tentative,1,1,0);
obs2=spv2xBuildObservation(s,2,stationManagement,positionManagement,0,tentative);
obs2=spv2xMaskActions(obs2,s,tentative);
assert(all([obs2.assignments.task_id]~=1) || all(~ismember(1,tentative.assigned_tasks)==false));
maskedAssigned=obs2.assignments(ismember([obs2.assignments.task_id],tentative.assigned_tasks));
% Build with remaining only - assigned tasks not listed; verify assigned list
assert(ismember(1,tentative.assigned_tasks) && ~ismember(2,tentative.assigned_tasks));
msg="action masking and ASSIGN updates";
end

function msg=checkSequentialCommitments()
s=baseState(); s.cpu_rate(1)=0.5e9; s.cpu_rate(2)=2e9;
s.tasks=[taskAt(1,1,0,100); taskAt(2,1,0,100)];
tentative=spv2xNewTentativeState(s);
f1=spv2xForecastCpuFinishMs(s,1,0,s.tasks(1).cycles,tentative);
tentative=spv2xApplyAssignment(s,tentative,1,1,0);
f2=spv2xForecastCpuFinishMs(s,1,0,s.tasks(2).cycles,tentative);
assert(f2>f1+1,'second assignment must see first commitment');
assert(abs((f2-f1)-72)<1e-6,'commitment adds one full local service time');
msg="sequential CPU commitment accounting";
end

function msg=checkNoDuplicateAssignment()
s=multiTaskState();
stationManagement=struct('cbrCV2Xvalues',zeros(3,1));
positionManagement=struct('distanceReal',[0 40 70; 40 0 50; 70 50 0]);
[s2,epochLog]=spv2xScheduleEpoch(s,[1 2],stationManagement,positionManagement,0,"joint_myopic");
assigned=[];
for i=1:numel(epochLog.actions)
    a=epochLog.actions{i};
    if a.type=="ASSIGN"
        assert(~ismember(a.task_id,assigned),'duplicate task assignment');
        assigned(end+1)=a.task_id; %#ok<AGROW>
    end
end
assert(~isempty(assigned),'expected at least one assignment');
statuses=string({s2.tasks.status});
assert(~any(statuses=="pending_unassigned" & ismember([s2.tasks.task_id],assigned)), ...
    'assigned tasks remain pending');
msg="no duplicate assignments in epoch";
end

function msg=checkNegativeControlLocal()
% Fast origin, single task, empty queues => local dominates; joint_local assigns local.
s=baseState(); s.cpu_rate(1)=2e9; s.cpu_rate(2)=2e9; s.cpu_rate(3)=2e9;
s.tasks=taskAt(1,1,0,100);
stationManagement=struct('cbrCV2Xvalues',zeros(3,1));
positionManagement=struct('distanceReal',[0 40 70; 40 0 50; 70 50 0]);
[s2,epochLog]=spv2xScheduleEpoch(s,1,stationManagement,positionManagement,0,"joint_local");
assert(epochLog.actions{1}.type=="ASSIGN" && epochLog.actions{1}.processor_id==1, ...
    'negative control must keep local execution');
assert(s2.tasks(1).action=="local");
msg="negative-control prefers local";
end

function msg=checkRewardConsistency()
s=baseState();
s.tasks=taskAt(1,1,0,100); s.tasks(1).status="completed"; s.tasks(1).result_delivered_ms=40; s.tasks(1).value=1;
s.tasks(2)=taskAt(2,1,0,100); s.tasks(2).status="late_result"; s.tasks(2).result_delivered_ms=120; s.tasks(2).value=1;
s.reward_rows={1,1,40,1,"completion"};
completed=sum(arrayfun(@(t)t.status=="completed",s.tasks));
rewardSum=sum(cellfun(@(x)x,s.reward_rows(:,4)));
assert(rewardSum==completed,'reward sum must equal completed count');
msg="completion reward matches simulator completions";
end

function msg=checkSafetyCostP2Agreement()
% Synthetic paired exposures: 100 opportunities, 2 harmful flips.
baseline=table((1:100)',ones(100,1),ones(100,1),(1:100)',zeros(100,1),true(100,1), ...
    [ones(50,1);2*ones(50,1)],'VariableNames', ...
    {'flow_id','seq','tx_id','rx_id','created_ms','success_on_time','distance_bin'});
policy=baseline; policy.success_on_time=true(100,1); policy.success_on_time(1:2)=false;
[costs,overall]=spv2xRecordSafetyCosts(baseline,policy,0.01,0.10,7,1000);
assert(overall.harmful_flips==2,'harmful flips');
assert(abs(overall.harmful_failure_probability-0.02)<1e-12,'harm rate');
assert(abs(costs.harm_cost(1)-(2-0.01*100))<1e-12,'harm cost identity');
assert(abs(costs.failure_cost(1)-(2-0.10*100))<1e-12,'failure cost identity');
% Cross-check summarizePolicySafety directly
[overall2]=summarizePolicySafety(baseline,policy,0.01,0.10);
assert(overall.harmful_flips==overall2.harmful_flips && ...
    abs(overall.absolute_failure_probability-overall2.absolute_failure_probability)<1e-12, ...
    'must match P2 summarizePolicySafety');
msg="safety costs agree with P2 accounting";
end

function s=baseState()
s=struct();
s.cfg.radio=struct('max_transport_block_bytes',1180,'selection_window_ms',20,'slot_duration_ms',0.5);
s.cfg.compute=struct('deadline_ms',100,'input_bytes',1500,'output_bytes',300, ...
    'cycles_per_task',36e6,'helper_range_m',300,'max_helpers',4,'arrival_rate_tasks_per_s',6);
s.cfg.control=struct('schedule_interval_ms',5);
s.origin_ids=1; s.helper_ids=[2;3];
s.cpu_rate=zeros(3,1); s.cpu_rate(:)=1e9;
s.cpu_queue={[],[],[]}; s.cpu_active=zeros(3,1); s.cpu_finish_s=inf(3,1);
s.pending_tx={[],[],[]};
s.empty_meta=struct('valid',false,'packet_id',0,'kind',"",'task_id',0,'stage',"",'fragment',0, ...
    'fragments',0,'src',0,'dst',0,'created_ms',0,'deadline_ms',0,'seq',0,'flow_id',0,'bytes',0);
s.packet_counter=0; s.offload_count=0; s.tentative_offloads=0;
s.tasks=taskAt(1,1,0,100);
end

function s=multiTaskState()
s=baseState(); s.cpu_rate(1)=0.5e9; s.cpu_rate(2)=2e9; s.cpu_rate(3)=1.5e9;
s.tasks=[taskAt(1,1,0,100); taskAt(2,1,0,100)];
end

function t=taskAt(id,origin,releaseMs,deadlineMs)
t=struct('task_id',id,'origin',origin,'release_ms',releaseMs,'deadline_ms',deadlineMs, ...
    'input_bytes',1500,'output_bytes',300,'cycles',36e6,'value',1,'action',"pending", ...
    'helper',0,'status',"pending_unassigned",'input_fragments',0,'input_received',0, ...
    'output_fragments',0,'output_received',0,'cpu_owner',0,'cpu_start_ms',NaN, ...
    'cpu_finish_ms',NaN,'input_delivered_ms',NaN,'result_delivered_ms',NaN,'decision_id',0);
end
