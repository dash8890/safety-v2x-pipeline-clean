function runLocalOnlyAblation()
%RUNLOCALONLYABLATION LOCAL-ONLY commit ablation on seeds 1301 and 1302.
% Tests whether blocking REQUEST_REMOTE_COMMIT raises TU above 24.
% BC weights loaded from drl/outputs/fresh_candidate_bc.json (same as BC baseline).
root=fileparts(mfilename('fullpath'));
out=fullfile(root,'outputs','ablation_local_only');
cfg=fullfile(root,'config','p3_bc_selected.json');
seeds=[1301 1302];
results=struct();

for i=1:numel(seeds)
    s=seeds(i);
    casePath=fullfile(out,sprintf('seed%d',s));
    summary=runR3PilotCase(casePath,'workflow_local_only',s, ...
        'config',cfg,'cpu_offered_load',0.85, ...
        'fresh_ddqn_epsilon',0,'fresh_exploration_mode','none', ...
        'fresh_coverage_rate',0);
    tu=summary.total_value;
    results.(sprintf('seed%d',s))=tu;
    fprintf('seed%d  TU=%d\n',s,tu);
end

tu1301=results.seed1301;
tu1302=results.seed1302;
fprintf('\nLOCAL-ONLY ablation result:\n');
fprintf('  seed1301: TU=%d  (BC=24, DRL-b12=24)\n',tu1301);
fprintf('  seed1302: TU=%d  (BC=24, DRL-b12=23)\n',tu1302);
fprintf('  mean    : %.1f\n',(tu1301+tu1302)/2);
if (tu1301+tu1302)/2 > 24
    fprintf('  >> HEADROOM EXISTS: TU>24 is achievable via local-commit strategy\n');
elseif (tu1301+tu1302)/2 >= 23.5
    fprintf('  >> NO HEADROOM: local-only matches or slightly trails BC\n');
else
    fprintf('  >> LOCAL-ONLY HURTS: remote commit is load-bearing for some workflows\n');
end

fid=fopen(fullfile(out,'ablation_result.json'),'w');
fprintf(fid,'%s',jsonencode(struct('seed1301',tu1301,'seed1302',tu1302, ...
    'mean',(tu1301+tu1302)/2,'bc_mean',24,'note','local_only_bc_weights'), ...
    'PrettyPrint',true));
fclose(fid);
end
