function summaries=runP4ProbeSuite(caseRoot,policy,varargin)
%RUNP4PROBESUITE Frozen greedy development probes for convergence evidence.
arguments
    caseRoot (1,1) string
    policy (1,1) string = "workflow_fresh_ddqn"
end
arguments (Repeating)
    varargin
end
root=fileparts(mfilename('fullpath'));
cfg=fullfile(root,'config','p3_bc_selected.json');
weights="";
for i=1:2:numel(varargin)
    if string(varargin{i})=="config", cfg=string(varargin{i+1}); end
    if string(varargin{i})=="fresh_weights_json", weights=string(varargin{i+1}); end
end
% Reserved development probes: never used by exploratory rollout collection
% or the final held-out test.
seeds=[1301 1302 1303 1304 1305];
summaries=cell(numel(seeds),1);
for i=1:numel(seeds)
    summaries{i}=runR3PilotCase(fullfile(caseRoot,sprintf('seed%d',seeds(i))), ...
        policy,seeds(i),'config',cfg,'cpu_offered_load',0.85, ...
        'fresh_ddqn_epsilon',0,'fresh_exploration_mode','none', ...
        'fresh_coverage_rate',0,'fresh_weights_json',weights);
end
fid=fopen(fullfile(caseRoot,'probe_manifest.json'),'w');
fprintf(fid,'%s',jsonencode(struct('seeds',seeds,'policy',policy, ...
    'mode','greedy_exploit','config',cfg,'fresh_weights_json',weights),'PrettyPrint',true)); fclose(fid);
end
