function summary = runWiLabSafetyValidation(configPath,durationS,seeds)
%RUNWILABSAFETYVALIDATION Native safety-only PRR/CBR/loss validation.
arguments
    configPath (1,1) string = "config/literature_grounded.json"
    durationS (1,1) double = NaN
    seeds double = []
end
root=fileparts(mfilename('fullpath')); wilabRoot=fileparts(root); addpath(genpath(wilabRoot));
if ~isfile(configPath), configPath=fullfile(root,configPath); end
cfg=loadConfig(configPath);
if isnan(durationS), durationS=cfg.experiment.duration_s; end
if isempty(seeds), seeds=cfg.experiment.seeds(:)'; end
stamp=string(datetime('now','Format','yyyyMMdd_HHmmss'));
runRoot=fullfile(root,cfg.experiment.output_root,"safety_validation_"+stamp);
if ~isfolder(runRoot), mkdir(runRoot); end
copyfile(configPath,fullfile(runRoot,"config.json"));
cfgFile=fullfile(wilabRoot,"ConfigFiles","mode2_pipeline.cfg");
densities=cfg.mobility.density_vehicles_per_km(:)'; rows=cell(0,13); scenario=0;
for density=densities
    densityIndex=find(densities==density,1);
    speed=cfg.mobility.speed_kmh_mean(min(densityIndex,numel(cfg.mobility.speed_kmh_mean)));
    speedStd=cfg.mobility.speed_kmh_std(min(densityIndex,numel(cfg.mobility.speed_kmh_std)));
    for seed=seeds
        scenario=scenario+1; outDir=fullfile(runRoot,sprintf("density_%03d_seed_%04d",density,seed));
        WiLabV2Xsim_final_policy_90(cfgFile,'Technology','NR-V2X','seed',seed, ...
            'simulationTime',durationS,'rho',density,'roadLength',cfg.mobility.road_length_m, ...
            'vMean',speed,'vStDev',speedStd,'BwMHz',cfg.radio.bandwidth_mhz, ...
            'SCS_NR',cfg.radio.subcarrier_spacing_khz,'MCS_NR',cfg.radio.mcs_index, ...
            'sizeSubchannel',cfg.radio.resource_blocks_per_subchannel, ...
            'beaconSizeBytes',cfg.safety.packet_bytes, ...
            'generationInterval',cfg.safety.generation_interval_ms/1000, ...
            'allocationPeriod',cfg.radio.reservation_interval_ms/1000, ...
            'probResKeep',cfg.radio.sps_keep_probability,'cv2xNumberOfReplicasMax',1, ...
            'Raw',cfg.safety.receiver_range_m,'dcc_active',false,'cbrActive',true, ...
            'printCBR',false,'printPacketReceptionRatio',true,'outputFolder',outDir, ...
            'safetyOffloadingEnabled',true,'safetyOffloadingConfig',configPath, ...
            'safetyOffloadingPolicy','no_offload','safetyOffloadingWorld','baseline');
        prrFiles=dir(fullfile(outDir,"packet_reception_ratio_*_5G.xls"));
        assert(~isempty(prrFiles),"Validation:MissingPRR","Native PRR output is missing");
        prrRaw=readmatrix(fullfile(prrFiles(1).folder,prrFiles(1).name), ...
            'FileType','text','Delimiter','\t');
        prrRaw=prrRaw(prrRaw(:,1)<=cfg.safety.receiver_range_m,:);
        opportunities=sum(prrRaw(:,5)); correct=sum(prrRaw(:,2));
        phyErrors=sum(prrRaw(:,3)); blocked=sum(prrRaw(:,4));
        cbrData=readmatrix(fullfile(outDir,"DATAvalues.xlsx")); cbr=cbrData(1);
        if ~isfinite(cbr), cbr=localMeanCBR(outDir); end
        safety=readtable(fullfile(outDir,"safety_offloading","safety_packets.csv"));
        rows(end+1,:)={scenario,seed,density,round(density*cfg.mobility.road_length_m/1000), ...
            durationS,height(safety),correct/opportunities,mean(safety.success_on_time),cbr, ...
            phyErrors/opportunities,blocked/opportunities,1-correct/opportunities,outDir}; %#ok<AGROW>
    end
end
summary=cell2table(rows,'VariableNames',{'scenario_id','seed','density_vehicles_per_km', ...
    'configured_vehicle_count','duration_s','logged_safety_exposures','native_prr','overlay_prr', ...
    'mean_cbr','phy_error_rate','half_duplex_blocking_rate','total_loss_rate','output_dir'});
writetable(summary,fullfile(runRoot,"safety_validation_summary.csv"));
save(fullfile(runRoot,"safety_validation_summary.mat"),'summary','cfg');
fprintf("Safety-only validation complete: %s\n",runRoot); disp(summary);
end

function value=localMeanCBR(outDir)
values=readmatrix(fullfile(outDir,"CBRvalues.xlsx"));
if size(values,2)>5, values=values(:,6:end); end
values=values(isfinite(values));
if isempty(values), value=NaN; else, value=mean(values); end
end
