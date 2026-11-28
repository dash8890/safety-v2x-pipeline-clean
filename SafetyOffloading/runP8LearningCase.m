function summary=runP8LearningCase(baseConfigPath,caseRoot,spec,baselineRoot,policyId)
%RUNP8LEARNINGCASE Run one P8 learning policy vs P4 baselines (P2 safety accounting).
arguments
    baseConfigPath (1,1) string
    caseRoot (1,1) string
    spec struct
    baselineRoot (1,1) string
    policyId (1,1) string
end
root=fileparts(mfilename('fullpath')); repoRoot=fileparts(root); addpath(genpath(repoRoot));
if ~isfile(baseConfigPath), baseConfigPath=fullfile(root,baseConfigPath); end
cfg=loadConfig(baseConfigPath); caseRoot=string(caseRoot); if ~isfolder(caseRoot), mkdir(caseRoot); end
cfg.mobility.density_vehicles_per_km=spec.density;
cfg.radio.selection_window_ms=spec.selection_window_slots*cfg.radio.slot_duration_ms;
cfg.compute.origin_fraction=0.10; cfg.compute.origin_cpu_ghz=spec.origin_cpu_ghz;
cfg.compute.helper_cpu_ghz=spec.helper_cpu_ghz;
cfg.compute.helper_background_utilization=spec.helper_background_utilization;
cfg.compute.arrival_model="poisson"; cfg.compute.arrival_rate_tasks_per_s=spec.arrival_rate;
cfg.compute.cycles_per_task=36000000; cfg.compute.deadline_ms=100;
cfg.compute.input_bytes=1500; cfg.compute.output_bytes=300;
cfg.safety.receiver_sample_fraction=0.10;
cfg.compute.task_start_ms=spec.warmup_s*1000;
cfg.compute.task_stop_ms=(spec.warmup_s+spec.observation_s)*1000;
cfg.control.risk_harm_limit=0.01;
cfg.control.schedule_interval_ms=5;
cfg.control.action_risk_model_path="models/p7_action_risk_calibrated.json";
cfg.control.completion_model_path="models/p7_completion_calibrated.json";
cfg.control.latency_model_path="models/p7_latency_calibrated.json";
cfg.control.p8_ddqn_model_path="models/p8_masked_ddqn.json";
cfg.control.p8_value_model_path="models/p8_value_assist.json";
cfg.control.p8_traj_model_path="models/p8_trajectory_constraint.json";
cfg.control.value_assist_lambda=0.25;
simulationTime=spec.warmup_s+spec.observation_s+0.2;
speed=max(70,110-0.4*(spec.density-50)); speedStd=max(7,11-0.04*(spec.density-50));

label=char(policyId);
runWorld(label,policyId,cfg);

baselineDir=fullfile(baselineRoot,sprintf('cfg%02d_seed%d',spec.configuration_id,spec.seed));
base=readWorld(fullfile(baselineDir,'deadline_local'));
safeDir="";
baseSummary=fullfile(baselineDir,'case_summary.csv');
if isfile(baseSummary)
    tab=readtable(baseSummary,'VariableNamingRule','preserve');
    for methodName=["safety_search","radio_search"]
        row=tab(string(tab.method)==methodName,:);
        if isempty(row), continue; end
        src=methodName;
        if ismember('source_world',row.Properties.VariableNames) && strlength(string(row.source_world(1)))>0
            src=string(row.source_world(1));
        end
        cand=fullfile(baselineDir,src);
        if isfolder(fullfile(cand,'safety_offloading')), safeDir=cand; break; end
    end
end
if strlength(safeDir)==0
    for src=["radio_search","safety_search"]
        cand=fullfile(baselineDir,src);
        if isfolder(fullfile(cand,'safety_offloading')), safeDir=cand; break; end
    end
end
if strlength(safeDir)==0
    error('runP8LearningCase:MissingSafetyBaseline','No safety baseline under %s',baselineDir);
end
safe=readWorld(safeDir);
pol=readWorld(fullfile(caseRoot,label));
% Also load calibrated myopic reference metrics if present for side-by-side.
p7Dir=fullfile(fileparts(caseRoot),'..','p7_calibrated_myopic_eval',sprintf('cfg%02d_seed%d',spec.configuration_id,spec.seed));
[overall,byStratum]=summarizePolicySafety(base.safety,pol.safety,0.01,0.10);
feasible=overall.harm_constraint_met && overall.failure_constraint_met && ...
    all(byStratum.harm_constraint_met & byStratum.failure_constraint_met);
nL=sum(base.tasks.completed_on_time); nM=sum(pol.tasks.completed_on_time); nS=sum(safe.tasks.completed_on_time);
headroom=(nS-nM)/max(1,nM);
nP7=NaN;
p7Sum=fullfile(string(p7Dir),'case_summary.csv');
if isfile(p7Sum)
    t7=readtable(p7Sum,'VariableNamingRule','preserve');
    if ismember('completed',t7.Properties.VariableNames), nP7=t7.completed(1); end
end
summary=table(string(policyId),nM,height(pol.tasks),nM/max(1,height(pol.tasks)), ...
    nM-nL,(nM-nL)/max(1,nL),sum(string(pol.tasks.action)=="offload"), ...
    overall.harmful_failure_probability,overall.absolute_failure_probability, ...
    logical(feasible),nL,nS,headroom,nP7, ...
    'VariableNames',{'method','completed','tasks','completion_ratio','completion_gain','relative_gain', ...
    'offloaded','harmful_failure_probability','absolute_safety_failure','harm_constraint_met', ...
    'local_completed','safety_search_completed','learning_headroom','calibrated_myopic_completed'});
writetable(summary,fullfile(caseRoot,'case_summary.csv'));
writetable(byStratum,fullfile(caseRoot,'safety_strata.csv'));
save(fullfile(caseRoot,'CASE_COMPLETE.mat'),'summary','spec','feasible','headroom');

    function runWorld(worldLabel,policy,worldCfg)
        worldRoot=fullfile(caseRoot,worldLabel); marker=fullfile(worldRoot,'COMPLETE.mat');
        if isfile(marker), return; end
        if ~isfolder(worldRoot), mkdir(worldRoot); end
        configPath=fullfile(worldRoot,'config.json');
        fid=fopen(configPath,'wt'); fprintf(fid,'%s',jsonencode(worldCfg,'PrettyPrint',true)); fclose(fid);
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
            'safetyOffloadingPolicy',char(policy),'safetyOffloadingWorld',worldLabel,'outputFolder',worldRoot};
        WiLabV2Xsim_final_policy_90(common{:}); completedAt=datetime('now'); save(marker,'completedAt','policy');
    end

    function world=readWorld(labelOrPath)
        if contains(string(labelOrPath),'seed') || contains(string(labelOrPath),'cfg')
            folder=fullfile(labelOrPath,'safety_offloading');
        else
            folder=fullfile(caseRoot,labelOrPath,'safety_offloading');
        end
        if ~isfolder(folder) && isfolder(fullfile(labelOrPath,'safety_offloading'))
            folder=fullfile(labelOrPath,'safety_offloading');
        end
        world.tasks=readtable(fullfile(folder,'tasks.csv'),'VariableNamingRule','preserve');
        world.safety=readtable(fullfile(folder,'safety_packets.csv'),'VariableNamingRule','preserve');
        keep=world.safety.created_ms>=spec.warmup_s*1000 & ...
            world.safety.created_ms<(spec.warmup_s+spec.observation_s)*1000;
        world.safety=world.safety(keep,:);
    end
end
