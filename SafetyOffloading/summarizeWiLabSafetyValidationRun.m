function summary=summarizeWiLabSafetyValidationRun(runRoot,configPath)
%SUMMARIZEWILABSAFETYVALIDATIONRUN Recover summaries from completed cases.
arguments
    runRoot (1,1) string
    configPath (1,1) string = "config/training_data.json"
end
root=fileparts(mfilename('fullpath'));
if ~isfolder(runRoot), runRoot=fullfile(root,runRoot); end
if ~isfile(configPath), configPath=fullfile(root,configPath); end
cfg=loadConfig(configPath); cases=dir(fullfile(runRoot,"density_*_seed_*")); rows=cell(0,13);
for k=1:numel(cases)
    outDir=fullfile(cases(k).folder,cases(k).name);
    if ~isfile(fullfile(outDir,"MainOut.xls")), continue; end
    tokens=sscanf(cases(k).name,"density_%d_seed_%d"); density=tokens(1); seed=tokens(2);
    prrFiles=dir(fullfile(outDir,"packet_reception_ratio_*_5G.xls"));
    prrRaw=readmatrix(fullfile(prrFiles(1).folder,prrFiles(1).name),'FileType','text','Delimiter','\t');
    prrRaw=prrRaw(prrRaw(:,1)<=cfg.safety.receiver_range_m,:);
    opportunities=sum(prrRaw(:,5)); correct=sum(prrRaw(:,2)); phyErrors=sum(prrRaw(:,3)); blocked=sum(prrRaw(:,4));
    data=readmatrix(fullfile(outDir,"DATAvalues.xlsx")); cbr=data(1);
    safety=readtable(fullfile(outDir,"safety_offloading","safety_packets.csv"));
    duration=max(safety.created_ms)/1000;
    rows(end+1,:)={k,seed,density,round(density*cfg.mobility.road_length_m/1000),duration, ...
        height(safety),correct/opportunities,mean(safety.success_on_time),cbr,phyErrors/opportunities, ...
        blocked/opportunities,1-correct/opportunities,outDir}; %#ok<AGROW>
end
summary=cell2table(rows,'VariableNames',{'scenario_id','seed','density_vehicles_per_km', ...
    'configured_vehicle_count','duration_s','logged_safety_exposures','native_prr','overlay_prr', ...
    'mean_cbr','phy_error_rate','half_duplex_blocking_rate','total_loss_rate','output_dir'});
writetable(summary,fullfile(runRoot,"safety_validation_summary.csv"));
save(fullfile(runRoot,"safety_validation_summary.mat"),'summary','cfg'); disp(summary);
end
