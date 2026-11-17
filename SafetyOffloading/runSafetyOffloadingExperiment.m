function summary = runSafetyOffloadingExperiment(configPath)
%RUNSAFETYOFFLOADINGEXPERIMENT Execute paired worlds and write contract logs.
arguments
    configPath (1,1) string = "config/literature_grounded.json"
end
root = fileparts(mfilename('fullpath'));
if ~isfile(configPath), configPath = fullfile(root,configPath); end
cfg = loadConfig(configPath);
stamp = string(datetime('now','Format','yyyyMMdd_HHmmss'));
runDir = fullfile(root,cfg.experiment.output_root,cfg.project.profile+"_"+stamp);
if ~isfolder(runDir), mkdir(runDir); end
copyfile(configPath,fullfile(runDir,"config.json"));

densities = cfg.mobility.density_vehicles_per_km(:)';
seeds = cfg.experiment.seeds(:)';
policies = string(cfg.experiment.policies(:)');
summaryRows = cell(0,12);
scenarioRows = cell(0,9);
scenarioId = 0;

for density = densities
    for seed = seeds
        scenarioId = scenarioId + 1;
        trace = generateScenario(cfg,density,seed);
        baseline = simulateWorld(cfg,trace,"no_offload");
        baseDir = fullfile(runDir,sprintf("scenario_%04d",scenarioId),"no_offload");
        writeWorld(baseDir,scenarioId,"baseline",baseline);
        for policy = policies
            treatment = simulateWorld(cfg,trace,policy);
            paired = pairOutcomes(baseline.safety,treatment.safety);
            policyDir = fullfile(runDir,sprintf("scenario_%04d",scenarioId),policy);
            writeWorld(policyDir,scenarioId,"policy",treatment);
            writetable(paired,fullfile(policyDir,"paired_outcomes.csv"));
            n = height(paired);
            harm = sum(paired.harmful_flip)/max(1,n);
            failure = sum(~paired.policy_success)/max(1,n);
            baseFailure = sum(~paired.baseline_success)/max(1,n);
            avoided = sum(paired.avoided_failure)/max(1,n);
            completed = treatment.summary.tasks_completed;
            summaryRows(end+1,:) = {scenarioId,seed,density,char(policy),n,harm,failure,baseFailure, ...
                avoided,completed,treatment.summary.cbr,treatment.summary.fragments}; %#ok<AGROW>
            scenarioRows(end+1,:) = {scenarioId,seed,density,cfg.project.profile,"matlab-mode2-v1", ...
                "release16-profile",char(policy),cfg.radio.subchannels,cfg.experiment.duration_s}; %#ok<AGROW>
        end
    end
end

summary = cell2table(summaryRows,'VariableNames',{'scenario_id','seed','density','policy', ...
    'safety_opportunities','harmful_flip_probability','absolute_failure_probability', ...
    'baseline_failure_probability','avoided_failure_probability','tasks_completed', ...
    'cbr','fragment_count'});
scenarios = cell2table(scenarioRows,'VariableNames',{'scenario_id','seed','regime', ...
    'trace_id','sim_version','release','policy_id','pool_subchannels','duration_s'});
writetable(summary,fullfile(runDir,"summary.csv"));
writetable(scenarios,fullfile(runDir,"scenario.csv"));
save(fullfile(runDir,"summary.mat"),'summary','cfg');
fprintf("Safety-offloading run complete: %s\n",runDir);
disp(summary);
end

function writeWorld(folder,scenarioId,world,result)
if ~isfolder(folder), mkdir(folder); end
safety=result.safety; safety.scenario_id=repmat(scenarioId,height(safety),1);
safety.world=repmat(string(world),height(safety),1);
safety=movevars(safety,{'scenario_id','world'},'Before',1);
tasks=result.tasks; tasks.scenario_id=repmat(scenarioId,height(tasks),1);
tasks=movevars(tasks,'scenario_id','Before',1);
decisions=result.decisions; decisions.scenario_id(:)=scenarioId;
radio=result.radio; radio.scenario_id(:)=scenarioId;
writetable(safety,fullfile(folder,"safety_packets.csv"));
writetable(tasks,fullfile(folder,"tasks.csv"));
writetable(decisions,fullfile(folder,"decisions.csv"));
writetable(radio,fullfile(folder,"radio_events.csv"));
end

