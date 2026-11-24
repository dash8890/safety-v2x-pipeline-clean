function summaries=runP4ExplorationBatch(caseRoot,batchId,varargin)
%RUNP4EXPLORATIONBATCH Collect one fresh, coverage-directed WiLab batch.
arguments
    caseRoot (1,1) string
    batchId (1,1) double {mustBeInteger,mustBePositive}
end
arguments (Repeating)
    varargin
end
root=fileparts(mfilename('fullpath'));
cfg=fullfile(root,'config','p3_bc_selected.json');
for i=1:2:numel(varargin)
    if string(varargin{i})=="config", cfg=string(varargin{i+1}); end
end
% Development rollout seeds are disjoint from P4 probes (1301:1305) and
% are advanced in non-overlapping batches by the orchestrator.
seeds=(1401+(batchId-1)*10):(1410+(batchId-1)*10);
summaries=cell(numel(seeds),1);
for i=1:numel(seeds)
    summaries{i}=runR3PilotCase(fullfile(caseRoot,sprintf('seed%d',seeds(i))), ...
        'workflow_fresh_bc',seeds(i),'config',cfg,'cpu_offered_load',0.85, ...
        'fresh_ddqn_epsilon',0.15,'fresh_exploration_mode','coverage', ...
        'fresh_coverage_rate',0.55);
end
manifest=struct('batch_id',batchId,'seeds',seeds,'policy','workflow_fresh_bc', ...
    'epsilon',0.15,'coverage_rate',0.55,'config',cfg);
fid=fopen(fullfile(caseRoot,'batch_manifest.json'),'w');
fprintf(fid,'%s',jsonencode(manifest,'PrettyPrint',true)); fclose(fid);
end
