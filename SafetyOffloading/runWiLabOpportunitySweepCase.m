function summary=runWiLabOpportunitySweepCase(baseConfigPath,caseRoot,spec)
%RUNWILABOPPORTUNITYSWEEPCASE Execute one matched P4 opportunity scenario.
%
% Independent WiLab worlds (no aliased rows):
%   deadline_local  strong local reference and safety reference world
%   ideal_oracle    ideal-communication local-first rescue
%   radio_search    actual-radio local-first rescue, unconstrained admission
%   risk_myopic     provisional conditional-risk admission (see Tick)
%   safety_cand_*   budget-capped radio rescue at 10/25/50/75% of N_R, plus
%                   at most one refinement between the best feasible and the
%                   next higher evaluated budget
% The 0% and 100% search candidates are the already evaluated deadline_local
% and radio_search worlds.  safety_search is a *selection* among evaluated
% candidates (row_type "selection"), not an extra simulation.
arguments
    baseConfigPath (1,1) string
    caseRoot (1,1) string
    spec struct
end
root=fileparts(mfilename('fullpath')); repoRoot=fileparts(root); addpath(genpath(repoRoot));
if ~isfile(baseConfigPath), baseConfigPath=fullfile(root,baseConfigPath); end
cfg=loadConfig(baseConfigPath); caseRoot=string(caseRoot); if ~isfolder(caseRoot), mkdir(caseRoot); end
cfg.mobility.density_vehicles_per_km=spec.density;
cfg.radio.selection_window_ms=spec.selection_window_slots*cfg.radio.slot_duration_ms;
assert(cfg.radio.selection_window_ms==round(cfg.radio.selection_window_ms) && ...
    cfg.radio.selection_window_ms>=20 && cfg.radio.selection_window_ms<=100, ...
    'Sweep:SelectionWindow','WiLab V6.2 requires an integer T2 in the range 20-100 ms');
cfg.compute.origin_fraction=0.10; cfg.compute.origin_cpu_ghz=spec.origin_cpu_ghz;
cfg.compute.helper_cpu_ghz=spec.helper_cpu_ghz;
cfg.compute.helper_background_utilization=spec.helper_background_utilization;
cfg.compute.arrival_model="poisson"; cfg.compute.arrival_rate_tasks_per_s=spec.arrival_rate;
cfg.compute.cycles_per_task=36000000; cfg.compute.deadline_ms=100;
cfg.compute.input_bytes=1500; cfg.compute.output_bytes=300;
cfg.safety.receiver_sample_fraction=0.10;
if isfield(spec,'receiver_sample_fraction'), cfg.safety.receiver_sample_fraction=spec.receiver_sample_fraction; end
cfg.compute.task_start_ms=spec.warmup_s*1000;
cfg.compute.task_stop_ms=(spec.warmup_s+spec.observation_s)*1000;
simulationTime=spec.warmup_s+spec.observation_s+0.2;
speed=max(70,110-0.4*(spec.density-50)); speedStd=max(7,11-0.04*(spec.density-50));
harmLimit=0.01; failureLimit=0.10;

% ---- Independent reference / oracle / provisional-myopic worlds ----------
methods={"deadline_local","deadline_local",false; ...
    "ideal_oracle","oracle_instant",true; ...
    "radio_search","oracle_actual",false; ...
    "risk_myopic","risk_myopic",false};
for mi=1:size(methods,1)
    worldCfg=cfg; worldCfg.compute.ideal_communication=methods{mi,3};
    runWorld(methods{mi,1},methods{mi,2},worldCfg);
end
baseline=readWorld("deadline_local"); baseTasks=baseline.tasks;
baseCompleted=sum(baseTasks.completed_on_time);
radioWorld=readWorld("radio_search");
proposedOffloads=sum(string(radioWorld.tasks.action)=="offload");   % N_R

% ---- Expanded privileged safety search -----------------------------------
% candidates: struct array of label, fraction, budget, feasible, completed
candidates=struct('label',{"deadline_local","radio_search"},'fraction',{0,1}, ...
    'budget',{0,proposedOffloads},'feasible',{true,[]},'completed',{baseCompleted,[]});
candidates(2)=evaluateCandidate(candidates(2),"radio_search");
seenBudgets=[0 proposedOffloads];
for fraction=[0.10 0.25 0.50 0.75]
    budget=round(fraction*proposedOffloads);
    if budget<1 || ismember(budget,seenBudgets), continue; end
    candidates(end+1)=runBudget(fraction,budget); %#ok<AGROW>
    seenBudgets(end+1)=budget; %#ok<AGROW>
end
% One refinement only: midpoint between the best feasible budget and the
% next higher evaluated budget (non-monotone harm is possible, so this is
% a single intermediate probe, not a bisection).
feasibleMask=[candidates.feasible];
[~,bestIdx]=max(arrayfun(@(c)c.completed*c.feasible-(~c.feasible)*1e9,candidates));
fractions=[candidates.fraction]; higher=fractions(fractions>candidates(bestIdx).fraction);
refined=false;
if ~isempty(higher) && proposedOffloads>=2
    midFraction=(candidates(bestIdx).fraction+min(higher))/2;
    midBudget=round(midFraction*proposedOffloads);
    if midBudget>=1 && ~ismember(midBudget,seenBudgets)
        candidates(end+1)=runBudget(midFraction,midBudget); refined=true; %#ok<AGROW>
    end
end
% Accounting correction: N_S is the best demonstrated *feasible* schedule
% among every evaluated method, including provisional risk_myopic.  Ideal
% communication is excluded because it does not induce radio externality.
rmCand=struct('label',"risk_myopic",'fraction',NaN,'budget',NaN,'feasible',[],'completed',[]);
candidates(end+1)=evaluateCandidate(rmCand,"risk_myopic"); %#ok<AGROW>
score=arrayfun(@(c)c.completed*logical(c.feasible)-(~logical(c.feasible))*1e9,candidates);
[~,bestIdx]=max(score); best=candidates(bestIdx);

% ---- Reporting ------------------------------------------------------------
worldLabels=["deadline_local","ideal_oracle","radio_search","risk_myopic"];
rowTypes=["reference","independent_run","independent_run","independent_run"];
for ci=1:numel(candidates)
    if ~ismember(candidates(ci).label,worldLabels)
        worldLabels(end+1)=candidates(ci).label; rowTypes(end+1)="search_candidate"; %#ok<AGROW>
    end
end
strataAll=table(); classesAll=table(); rows=cell(numel(worldLabels)+1,29);
for li=1:numel(worldLabels)
    [rows(li,:),byStratum,classCounts]=summarizeWorld(worldLabels(li),rowTypes(li));
    strataAll=[strataAll;byStratum]; classesAll=[classesAll;classCounts]; %#ok<AGROW>
end
[rows(end,:),~,~]=summarizeWorld(best.label,"selection","safety_search");
summary=cell2table(rows,'VariableNames',{'method','source_world','row_type','policy','completed','tasks', ...
    'completion_ratio','completion_gain','relative_gain','offloaded','rescued_tasks','newly_lost_tasks', ...
    'direct_remote_rescues','indirect_local_rescues','harmful_failure_probability','absolute_safety_failure', ...
    'safety_prr','harm_constraint_met','failure_constraint_met','input_latency_p50_ms','input_latency_p95_ms', ...
    'result_latency_p50_ms','result_latency_p95_ms','class_I','class_II','class_III','class_IV', ...
    'budget_fraction','budget_offloads'});
summary.proposed_offloads_NR=repmat(proposedOffloads,height(summary),1);
summary.search_refined=repmat(refined,height(summary),1);
writetable(summary,fullfile(caseRoot,'case_summary.csv'));
writetable(strataAll,fullfile(caseRoot,'safety_strata.csv'));
writetable(classesAll,fullfile(caseRoot,'feasibility_classes.csv'));
bestSafetyLabel=best.label; %#ok<NASGU>
save(fullfile(caseRoot,'CASE_COMPLETE.mat'),'summary','spec','cfg','bestSafetyLabel','candidates');

    function c=runBudget(fraction,budget)
        label="safety_cand_K"+compose('%03d',budget);
        worldCfg=cfg; worldCfg.compute.ideal_communication=false;
        worldCfg.compute.safety_budget_offloads=budget;
        runWorld(label,"safety_budget",worldCfg);
        c=struct('label',label,'fraction',fraction,'budget',budget,'feasible',[],'completed',[]);
        c=evaluateCandidate(c,label);
    end

    function c=evaluateCandidate(c,label)
        world=readWorld(label); [overall,byStratum]=summarizePolicySafety(baseline.safety,world.safety,harmLimit,failureLimit);
        c.feasible=logical(overall.harm_constraint_met && overall.failure_constraint_met && ...
            all(byStratum.harm_constraint_met & byStratum.failure_constraint_met));
        c.completed=sum(world.tasks.completed_on_time);
    end

    function [row,byStratum,classCounts]=summarizeWorld(label,rowType,methodName)
        if nargin<3, methodName=label; end
        world=readWorld(label); tasks=world.tasks; decisions=world.decisions;
        [overall,byStratum]=summarizePolicySafety(baseline.safety,world.safety,harmLimit,failureLimit);
        byStratum.method=repmat(string(methodName),height(byStratum),1); byStratum=movevars(byStratum,'method','Before',1);
        byStratum.row_type=repmat(string(rowType),height(byStratum),1);
        classCounts=groupcounts(decisions,'feasibility_class');
        classCounts.method=repmat(string(methodName),height(classCounts),1); classCounts=movevars(classCounts,'method','Before',1);
        classCounts.row_type=repmat(string(rowType),height(classCounts),1);
        completed=sum(tasks.completed_on_time); offloaded=sum(string(tasks.action)=="offload");
        [rescued,lost,direct,indirect]=rescueCounts(baseTasks,tasks);
        off=tasks(string(tasks.action)=="offload",:);
        [inputP50,inputP95]=percentiles(off.input_delivered_ms-off.release_ms);
        [resultP50,resultP95]=percentiles(off.result_delivered_ms-off.cpu_finish_ms);
        classN=zeros(1,4); classLabels=["I","II","III","IV"];
        for k=1:4, classN(k)=sum(string(decisions.feasibility_class)==classLabels(k)); end
        idx=find([candidates.label]==label,1); fraction=NaN; budget=NaN;
        if ~isempty(idx), fraction=candidates(idx).fraction; budget=candidates(idx).budget; end
        policyId=string(decisions.policy_id(1));
        row={string(methodName),string(label),string(rowType),policyId,completed,height(tasks), ...
            completed/max(1,height(tasks)),completed-baseCompleted,(completed-baseCompleted)/max(1,baseCompleted), ...
            offloaded,rescued,lost,direct,indirect,overall.harmful_failure_probability, ...
            overall.absolute_failure_probability,overall.safety_prr,overall.harm_constraint_met, ...
            overall.failure_constraint_met,inputP50,inputP95,resultP50,resultP95, ...
            classN(1),classN(2),classN(3),classN(4),fraction,budget};
    end

    function runWorld(label,policy,worldCfg)
        worldRoot=fullfile(caseRoot,label); marker=fullfile(worldRoot,'COMPLETE.mat');
        if isfile(marker), return; end
        if ~isfolder(worldRoot), mkdir(worldRoot); end
        configPath=fullfile(worldRoot,'config.json');
        fid=fopen(configPath,'wt'); assert(fid>=0,'Sweep:Config','Cannot write case config');
        fprintf(fid,'%s',jsonencode(worldCfg,'PrettyPrint',true)); fclose(fid);
        cfgFile=fullfile(repoRoot,'ConfigFiles','mode2_pipeline.cfg');
        common={cfgFile,'Technology','NR-V2X','seed',spec.seed,'simulationTime',simulationTime, ...
            'rho',spec.density,'roadLength',worldCfg.mobility.road_length_m,'vMean',speed,'vStDev',speedStd, ...
            'BwMHz',worldCfg.radio.bandwidth_mhz,'SCS_NR',worldCfg.radio.subcarrier_spacing_khz, ...
            'MCS_NR',worldCfg.radio.mcs_index,'sizeSubchannel',worldCfg.radio.resource_blocks_per_subchannel, ...
            'beaconSizeBytes',worldCfg.radio.max_transport_block_bytes, ...
            'generationInterval',worldCfg.safety.generation_interval_ms/1000, ...
            'allocationPeriod',worldCfg.radio.reservation_interval_ms/1000, ...
            'T1autonomousMode',worldCfg.radio.slot_duration_ms, ...
            'T2autonomousMode',worldCfg.radio.selection_window_ms, ...
            'TsensingPeriod',worldCfg.radio.sensing_window_ms/1000, ...
            'ratioSelectedAutonomousMode',worldCfg.radio.candidate_fraction, ...
            'powerThresholdAutonomous',worldCfg.radio.sensing_threshold_dbm, ...
            'probResKeep',worldCfg.radio.sps_keep_probability,'cv2xNumberOfReplicasMax',1, ...
            'Raw',worldCfg.radio.native_wilab_raw_m,'dcc_active',false,'cbrActive',true,'printCBR',false, ...
            'safetyOffloadingEnabled',true,'safetyOffloadingConfig',configPath, ...
            'safetyOffloadingPolicy',policy,'safetyOffloadingWorld',label,'outputFolder',worldRoot};
        WiLabV2Xsim_final_policy_90(common{:}); completedAt=datetime('now'); save(marker,'completedAt','policy');
    end

    function world=readWorld(label)
        folder=fullfile(caseRoot,label,'safety_offloading');
        world.tasks=readtable(fullfile(folder,'tasks.csv'),'VariableNamingRule','preserve');
        world.safety=readtable(fullfile(folder,'safety_packets.csv'),'VariableNamingRule','preserve');
        world.decisions=readtable(fullfile(folder,'decisions.csv'),'VariableNamingRule','preserve');
        keep=world.safety.created_ms>=spec.warmup_s*1000 & ...
            world.safety.created_ms<(spec.warmup_s+spec.observation_s)*1000;
        world.safety=world.safety(keep,:);
    end
end

function [rescued,lost,direct,indirect]=rescueCounts(baseTasks,tasks)
base=baseTasks(:,{'task_id','completed_on_time'}); base.Properties.VariableNames{2}='base_completed';
policy=tasks(:,{'task_id','completed_on_time','action'}); policy.Properties.VariableNames{2}='policy_completed';
joined=innerjoin(base,policy,'Keys','task_id'); rescue=~joined.base_completed & joined.policy_completed;
lostMask=joined.base_completed & ~joined.policy_completed; remote=rescue & string(joined.action)=="offload";
rescued=sum(rescue); lost=sum(lostMask); direct=sum(remote); indirect=sum(rescue & ~remote);
end

function [p50,p95]=percentiles(values)
values=sort(values(isfinite(values))); if isempty(values), p50=NaN; p95=NaN; return; end
p50=values(max(1,ceil(0.50*numel(values)))); p95=values(max(1,ceil(0.95*numel(values))));
end
