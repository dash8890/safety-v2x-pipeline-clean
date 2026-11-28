function summary=runP9InfoDiagnosticCase(baseConfigPath,caseRoot,spec,baselineRoot,p7Root)
%RUNP9INFODIAGNOSTICCASE Privileged vs deployable search vs calibrated myopic.
% Simulates only oracle_deployable; reuses P4 radio_search / safety_search and P7.
arguments
    baseConfigPath (1,1) string
    caseRoot (1,1) string
    spec struct
    baselineRoot (1,1) string
    p7Root (1,1) string = ""
end
root=fileparts(mfilename('fullpath')); repoRoot=fileparts(root); addpath(genpath(repoRoot));
if ~isfile(baseConfigPath), baseConfigPath=fullfile(root,baseConfigPath); end
if strlength(p7Root)==0
    p7Root=fullfile(root,"outputs","p7_calibrated_myopic_eval");
end
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
simulationTime=spec.warmup_s+spec.observation_s+0.2;
speed=max(70,110-0.4*(spec.density-50)); speedStd=max(7,11-0.04*(spec.density-50));

runWorld("oracle_deployable","oracle_deployable",cfg);

baselineDir=fullfile(baselineRoot,sprintf('cfg%02d_seed%d',spec.configuration_id,spec.seed));
base=readWorldFolder(fullfile(baselineDir,'deadline_local'));
radio=readWorldFolder(fullfile(baselineDir,'radio_search'));
safeDir=resolveSafetyDir(baselineDir);
safe=readWorldFolder(safeDir);
dep=readWorldFolder(fullfile(caseRoot,'oracle_deployable'));
p7Dir=fullfile(p7Root,sprintf('cfg%02d_seed%d',spec.configuration_id,spec.seed),'risk_myopic_p7');
p7=readWorldFolder(p7Dir);

nL=sum(base.tasks.completed_on_time);
nR=sum(radio.tasks.completed_on_time);
nS=sum(safe.tasks.completed_on_time);
nD=sum(dep.tasks.completed_on_time);
nP=sum(p7.tasks.completed_on_time);
oR=sum(string(radio.tasks.action)=="offload");
oD=sum(string(dep.tasks.action)=="offload");
oP=sum(string(p7.tasks.action)=="offload");

[ovR]=summarizePolicySafety(base.safety,radio.safety,0.01,0.10);
[ovS]=summarizePolicySafety(base.safety,safe.safety,0.01,0.10);
[ovD]=summarizePolicySafety(base.safety,dep.safety,0.01,0.10);
[ovP]=summarizePolicySafety(base.safety,p7.safety,0.01,0.10);

% Gap fractions relative to local.
gR=(nR-nL)/max(1,nL); gS=(nS-nL)/max(1,nL); gD=(nD-nL)/max(1,nL); gP=(nP-nL)/max(1,nL);
% Share of privileged radio_search gain retained under deployable forecasts.
if gR>1e-9, retain=gD/gR; else, retain=NaN; end
% Residual gap after removing privilege (radio privileged vs deployable).
infoGap=gR-gD;
% Residual gap after deployable search vs calibrated myopic (optimization / risk).
optGap=gD-gP;

summary=table(spec.configuration_id,spec.seed,nL,nR,nS,nD,nP,oR,oD,oP, ...
    gR,gS,gD,gP,retain,infoGap,optGap, ...
    ovR.harm_constraint_met,ovS.harm_constraint_met,ovD.harm_constraint_met,ovP.harm_constraint_met, ...
    ovR.harmful_failure_probability,ovD.harmful_failure_probability,ovP.harmful_failure_probability, ...
    'VariableNames',{'configuration_id','seed','local_completed', ...
    'radio_priv_completed','safety_search_completed','oracle_deployable_completed','p7_completed', ...
    'radio_priv_offloads','deployable_offloads','p7_offloads', ...
    'radio_priv_gain','safety_search_gain','deployable_gain','p7_gain', ...
    'deployable_retains_radio_gain','info_gap_radio_minus_deployable','opt_gap_deployable_minus_p7', ...
    'radio_harm_ok','safety_search_harm_ok','deployable_harm_ok','p7_harm_ok', ...
    'radio_harm_rate','deployable_harm_rate','p7_harm_rate'});
writetable(summary,fullfile(caseRoot,'case_summary.csv'));
save(fullfile(caseRoot,'CASE_COMPLETE.mat'),'summary','spec');

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

    function world=readWorldFolder(folder)
        so=fullfile(folder,'safety_offloading');
        if ~isfolder(so), error('P9Diag:MissingWorld','Missing %s',so); end
        world.tasks=readtable(fullfile(so,'tasks.csv'),'VariableNamingRule','preserve');
        world.safety=readtable(fullfile(so,'safety_packets.csv'),'VariableNamingRule','preserve');
        keep=world.safety.created_ms>=spec.warmup_s*1000 & ...
            world.safety.created_ms<(spec.warmup_s+spec.observation_s)*1000;
        world.safety=world.safety(keep,:);
    end
end

function safeDir=resolveSafetyDir(baselineDir)
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
    error('P9Diag:MissingSafety','No safety/radio world under %s',baselineDir);
end
end
