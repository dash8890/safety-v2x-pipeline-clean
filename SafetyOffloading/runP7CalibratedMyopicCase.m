function summary=runP7CalibratedMyopicCase(baseConfigPath,caseRoot,spec,baselineRoot)
%RUNP7CALIBRATEDMYOPICCASE Run risk_myopic_p7 with P7 calibrated estimators.
% Reuses P4 local / safety baselines. P6 worlds remain frozen and unused here.
arguments
    baseConfigPath (1,1) string
    caseRoot (1,1) string
    spec struct
    baselineRoot (1,1) string
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
if isfield(spec,'receiver_sample_fraction'), cfg.safety.receiver_sample_fraction=spec.receiver_sample_fraction; end
cfg.compute.task_start_ms=spec.warmup_s*1000;
cfg.compute.task_stop_ms=(spec.warmup_s+spec.observation_s)*1000;
cfg.control.risk_harm_limit=0.01;
cfg.control.schedule_interval_ms=5;
cfg.control.action_risk_model_path="models/p7_action_risk_calibrated.json";
cfg.control.completion_model_path="models/p7_completion_calibrated.json";
cfg.control.latency_model_path="models/p7_latency_calibrated.json";
simulationTime=spec.warmup_s+spec.observation_s+0.2;
speed=max(70,110-0.4*(spec.density-50)); speedStd=max(7,11-0.04*(spec.density-50));

runWorld("risk_myopic_p7","risk_myopic_p7",cfg);

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
    error('runP7CalibratedMyopicCase:MissingSafetyBaseline','No safety_search/radio_search world under %s',baselineDir);
end
safe=readWorld(safeDir);
myopic=readWorld(fullfile(caseRoot,'risk_myopic_p7'));
[overall,byStratum]=summarizePolicySafety(base.safety,myopic.safety,0.01,0.10);
feasible=overall.harm_constraint_met && overall.failure_constraint_met && ...
    all(byStratum.harm_constraint_met & byStratum.failure_constraint_met);
nL=sum(base.tasks.completed_on_time); nM=sum(myopic.tasks.completed_on_time); nS=sum(safe.tasks.completed_on_time);
headroom=(nS-nM)/max(1,nM);
summary=table("risk_myopic_p7",nM,height(myopic.tasks),nM/max(1,height(myopic.tasks)), ...
    nM-nL,(nM-nL)/max(1,nL),sum(string(myopic.tasks.action)=="offload"), ...
    overall.harmful_failure_probability,overall.absolute_failure_probability, ...
    logical(feasible),nL,nS,headroom, ...
    'VariableNames',{'method','completed','tasks','completion_ratio','completion_gain','relative_gain', ...
    'offloaded','harmful_failure_probability','absolute_safety_failure','harm_constraint_met', ...
    'local_completed','safety_search_completed','learning_headroom'});
writetable(summary,fullfile(caseRoot,'case_summary.csv'));
writetable(byStratum,fullfile(caseRoot,'safety_strata.csv'));
save(fullfile(caseRoot,'CASE_COMPLETE.mat'),'summary','spec','feasible','headroom');

    function runWorld(label,policy,worldCfg)
        worldRoot=fullfile(caseRoot,label); marker=fullfile(worldRoot,'COMPLETE.mat');
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
            'safetyOffloadingPolicy',policy,'safetyOffloadingWorld',label,'outputFolder',worldRoot};
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
