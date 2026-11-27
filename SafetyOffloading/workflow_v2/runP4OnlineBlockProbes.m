function runP4OnlineBlockProbes(updateBlock,runName)
%RUNP4ONLINEBLOCKPROBES Efficient fixed-seed validation at a block end.
arguments
    updateBlock (1,1) double {mustBeInteger,mustBePositive}
    runName (1,1) string = "p4_online80_dueling_run01"
end
root=fileparts(mfilename('fullpath'));
updateRoot=fullfile(root,'outputs',runName,'training',sprintf('update%02d',updateBlock));
weights=fullfile(updateRoot,'policy.json');
if ~isfile(weights), error('Missing block-end policy: %s',weights); end
seeds=[1301 1302];
out=fullfile(root,'outputs',runName,'validation',sprintf('block%02d',updateBlock));
cfg=fullfile(root,'config','p3_bc_selected.json');
for i=1:numel(seeds)
    runR3PilotCase(fullfile(out,sprintf('seed%d',seeds(i))), ...
        'workflow_fresh_ddqn',seeds(i),'config',cfg,'cpu_offered_load',0.85, ...
        'fresh_ddqn_epsilon',0,'fresh_exploration_mode','none', ...
        'fresh_coverage_rate',0,'fresh_weights_json',weights);
end
fid=fopen(fullfile(out,'validation_manifest.json'),'w');
fprintf(fid,'%s',jsonencode(struct('block',updateBlock,'seeds',seeds, ...
    'weights',weights,'mode','greedy_fixed_validation'),'PrettyPrint',true)); fclose(fid);
end
