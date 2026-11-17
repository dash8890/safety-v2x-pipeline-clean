function [summary,trainingData] = runWiLabPairedExternality(configPath,durationS,seeds,treatmentPolicy)
%RUNWILABPAIREDEXTERNALITY Generate matched baseline/unrestricted risk data.
arguments
    configPath (1,1) string = "config/literature_grounded.json"
    durationS (1,1) double = NaN
    seeds double = []
    treatmentPolicy (1,1) string = "unrestricted"
end
root=fileparts(mfilename('fullpath')); wilabRoot=fileparts(root); addpath(genpath(wilabRoot));
if ~isfile(configPath), configPath=fullfile(root,configPath); end
cfg=loadConfig(configPath);
if isnan(durationS), durationS=cfg.experiment.duration_s; end
if isempty(seeds), seeds=cfg.experiment.seeds(:)'; end
stamp=string(datetime('now','Format','yyyyMMdd_HHmmss'));
runRoot=fullfile(root,cfg.experiment.output_root,"paired_"+treatmentPolicy+"_"+stamp);
if ~isfolder(runRoot), mkdir(runRoot); end
copyfile(configPath,fullfile(runRoot,"config.json"));
cfgFile=fullfile(wilabRoot,"ConfigFiles","mode2_pipeline.cfg");
densities=cfg.mobility.density_vehicles_per_km(:)'; summaryRows=cell(0,14);
allTraining=table(); scenario=0;
for density=densities
    densityIndex=find(densities==density,1);
    speed=cfg.mobility.speed_kmh_mean(min(densityIndex,numel(cfg.mobility.speed_kmh_mean)));
    speedStd=cfg.mobility.speed_kmh_std(min(densityIndex,numel(cfg.mobility.speed_kmh_std)));
    for seed=seeds
        scenario=scenario+1; scenarioRoot=fullfile(runRoot,sprintf("scenario_%04d",scenario));
        common={cfgFile,'Technology','NR-V2X','seed',seed,'simulationTime',durationS, ...
            'rho',density,'roadLength',cfg.mobility.road_length_m,'vMean',speed,'vStDev',speedStd, ...
            'BwMHz',cfg.radio.bandwidth_mhz,'SCS_NR',cfg.radio.subcarrier_spacing_khz, ...
            'MCS_NR',cfg.radio.mcs_index,'sizeSubchannel',cfg.radio.resource_blocks_per_subchannel, ...
            'beaconSizeBytes',cfg.radio.max_transport_block_bytes, ...
            'generationInterval',cfg.safety.generation_interval_ms/1000, ...
            'allocationPeriod',cfg.radio.reservation_interval_ms/1000, ...
            'probResKeep',cfg.radio.sps_keep_probability,'cv2xNumberOfReplicasMax',1, ...
            'Raw',cfg.radio.native_wilab_raw_m,'dcc_active',false,'cbrActive',true, ...
            'printCBR',false,'safetyOffloadingEnabled',true,'safetyOffloadingConfig',configPath};
        baseOut=fullfile(scenarioRoot,"no_offload"); treatmentOut=fullfile(scenarioRoot,treatmentPolicy);
        WiLabV2Xsim_final_policy_90(common{:},'outputFolder',baseOut, ...
            'safetyOffloadingPolicy','no_offload','safetyOffloadingWorld','baseline');
        WiLabV2Xsim_final_policy_90(common{:},'outputFolder',treatmentOut, ...
            'safetyOffloadingPolicy',treatmentPolicy,'safetyOffloadingWorld','policy');
        B=readtable(fullfile(baseOut,"safety_offloading","safety_packets.csv"),'VariableNamingRule','preserve');
        P=readtable(fullfile(treatmentOut,"safety_offloading","safety_packets.csv"),'VariableNamingRule','preserve');
        paired=pairOutcomes(B,P); paired.scenario_id=repmat(scenario,height(paired),1);
        paired.seed=repmat(seed,height(paired),1); paired.density=repmat(density,height(paired),1);
        paired=movevars(paired,{'scenario_id','seed','density'},'Before',1);
        writetable(paired,fullfile(scenarioRoot,"paired_safety.csv"));
        decisions=readtable(fullfile(treatmentOut,"safety_offloading","decisions.csv"));
        decisions.scenario_id(:)=scenario; decisions.seed=repmat(seed,height(decisions),1);
        decisions.density=repmat(density,height(decisions),1);
        tasks=readtable(fullfile(treatmentOut,"safety_offloading","tasks.csv"));
        training=labelDecisions(decisions,paired,tasks,cfg.safety.deadline_ms,treatmentPolicy);
        writetable(training,fullfile(scenarioRoot,"risk_training_rows.csv"));
        if isempty(allTraining), allTraining=training; else, allTraining=[allTraining;training]; end %#ok<AGROW>
        n=height(paired); summaryRows(end+1,:)={scenario,seed,density,durationS,n, ...
            mean(paired.baseline_success),mean(paired.policy_success),mean(paired.harmful_flip), ...
            mean(paired.avoided_failure),sum(paired.harmful_flip),sum(tasks.completed_on_time), ...
            sum(tasks.fragment_count),height(training),scenarioRoot}; %#ok<AGROW>
    end
end
summary=cell2table(summaryRows,'VariableNames',{'scenario_id','seed','density','duration_s', ...
    'safety_opportunities','baseline_prr','unrestricted_prr','harm_probability', ...
    'avoided_failure_probability','harmful_flips','tasks_completed','fragment_count', ...
    'training_rows','output_dir'});
trainingData=allTraining;
writetable(summary,fullfile(runRoot,"paired_externality_summary.csv"));
writetable(trainingData,fullfile(runRoot,"risk_training_dataset.csv"));
save(fullfile(runRoot,"paired_externality.mat"),'summary','trainingData','cfg');
fprintf("Paired externality data complete: %s\n",runRoot); disp(summary);
end

function training=labelDecisions(decisions,paired,tasks,windowMs,treatmentPolicy)
if treatmentPolicy=="dqn_explore", training=decisions;
else, training=decisions(string(decisions.chosen_action)=="offload",:); end
training.window_exposures=zeros(height(training),1); training.harmful_flips=zeros(height(training),1);
training.harm_rate=zeros(height(training),1); training.any_harm=false(height(training),1);
training.completed_on_time=false(height(training),1); training.failure_reason=strings(height(training),1);
training.action_index=zeros(height(training),1);
for i=1:height(training)
    startMs=training.time_ms(i); mask=paired.created_ms>=startMs & paired.created_ms<startMs+windowMs;
    training.window_exposures(i)=sum(mask); training.harmful_flips(i)=sum(paired.harmful_flip(mask));
    if training.window_exposures(i)>0
        training.harm_rate(i)=training.harmful_flips(i)/training.window_exposures(i);
    end
    training.any_harm(i)=training.harmful_flips(i)>0;
    taskRow=tasks(tasks.task_id==training.task_id(i),:);
    if ~isempty(taskRow)
        training.completed_on_time(i)=taskRow.completed_on_time(1);
        training.failure_reason(i)=string(taskRow.failure_reason(1));
    end
    action=string(training.chosen_action(i));
    if action=="reject"
        training.action_index(i)=0;
    elseif action=="local"
        training.action_index(i)=1;
    else
        training.action_index(i)=1+training.helper_rank(i);
    end
end
end
