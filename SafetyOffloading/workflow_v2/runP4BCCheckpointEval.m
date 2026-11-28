function runP4BCCheckpointEval(outputDir)
%RUNP4BCCHECKPOINTEVAL Evaluate frozen BC checkpoint on fixed validation seeds.
% Runs greedy evaluation (epsilon=0) using the initial BC weights on seeds
% 1301 and 1302, identical conditions to block-milestone validation, to
% establish whether RL updates improved or degraded the initial policy.
arguments
    outputDir (1,1) string = ""
end
root = fileparts(mfilename('fullpath'));
weights = fullfile(root, 'drl', 'outputs', 'fresh_candidate_bc.json');
if ~isfile(weights)
    error('BC checkpoint not found: %s', weights);
end
if strlength(outputDir) == 0
    outputDir = fullfile(root, 'outputs', 'p4_bc_checkpoint_eval');
end
seeds = [1301 1302];
cfg = fullfile(root, 'config', 'p3_bc_selected.json');
for i = 1:numel(seeds)
    seedDir = fullfile(outputDir, sprintf('seed%d', seeds(i)));
    runR3PilotCase(seedDir, ...
        'workflow_fresh_ddqn', seeds(i), 'config', cfg, ...
        'cpu_offered_load', 0.85, ...
        'fresh_ddqn_epsilon', 0, ...
        'fresh_exploration_mode', 'none', ...
        'fresh_coverage_rate', 0, ...
        'fresh_weights_json', weights);
end
manifest = struct('checkpoint', 'fresh_candidate_bc', ...
    'weights', weights, 'seeds', seeds, ...
    'mode', 'greedy_fixed_validation', ...
    'cpu_offered_load', 0.85, ...
    'generated_utc', char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z''')));
fid = fopen(fullfile(outputDir, 'bc_eval_manifest.json'), 'w');
fprintf(fid, '%s', jsonencode(manifest, 'PrettyPrint', true));
fclose(fid);
fprintf('BC checkpoint eval complete: %s\n', outputDir);
end
