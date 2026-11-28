function results=runP4CheckpointSeedSweep(outputDir,weights,seeds,label)
%RUNP4CHECKPOINTSEEDSWEEP Greedily evaluate one checkpoint over fixed seeds.
% Results are stored per seed and in a JSON summary for comparison with BC.
arguments
    outputDir (1,1) string
    weights (1,1) string
    seeds (1,:) double {mustBeInteger,mustBePositive}
    label (1,1) string
end
root=fileparts(mfilename('fullpath'));
if ~isfile(weights), error('Checkpoint not found: %s',weights); end
if ~isfolder(outputDir), mkdir(outputDir); end
cfg=fullfile(root,'config','p3_bc_selected.json');
policy="workflow_fresh_ddqn";
if label=="bc"
    policy="workflow_fresh_bc";
end
rows=struct('seed',cell(1,numel(seeds)),'tu',cell(1,numel(seeds)), ...
    'n_success',cell(1,numel(seeds)),'n_workflows',cell(1,numel(seeds)));
for i=1:numel(seeds)
    seed=seeds(i);
    summary=runR3PilotCase(fullfile(outputDir,sprintf('seed%d',seed)), ...
        policy,seed,'config',cfg,'cpu_offered_load',0.85, ...
        'fresh_ddqn_epsilon',0,'fresh_exploration_mode','none', ...
        'fresh_coverage_rate',0,'fresh_weights_json',weights);
    rows(i)=struct('seed',seed,'tu',double(summary.total_value), ...
        'n_success',double(summary.n_success),'n_workflows',double(summary.n_workflows));
end
tu=[rows.tu];
results=struct('label',label,'weights',weights,'seeds',seeds,'results',rows, ...
    'mean_tu',mean(tu),'generated_utc',char(datetime('now','TimeZone','UTC', ...
    'Format','yyyy-MM-dd''T''HH:mm:ss''Z''')));
fid=fopen(fullfile(outputDir,'seed_sweep_results.json'),'w');
fprintf(fid,'%s',jsonencode(results,'PrettyPrint',true)); fclose(fid);
fprintf('%s seed sweep complete: mean TU %.3f (%s)\n',label,results.mean_tu,outputDir);
end
