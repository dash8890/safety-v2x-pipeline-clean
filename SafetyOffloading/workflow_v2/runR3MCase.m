function summary=runR3MCase(outRoot,policy,seed)
%RUNR3MCASE Single R3-M shard for parallel launch.
root=fileparts(mfilename('fullpath'));
addpath(genpath(root));
cfgPath=fullfile(root,'config','r3m_gate.json');
cfg=jsondecode(fileread(cfgPath));
simTime=double(cfg.experiment.duration_s);
policy=string(policy);
seed=double(seed);
if ~isfolder(outRoot), mkdir(outRoot); end
caseRoot=fullfile(outRoot,sprintf('seed%d_%s',seed,policy));
fprintf('R3-M shard seed=%d policy=%s\n',seed,policy);
summary=runR3PilotCase(caseRoot,policy,seed,'config',cfgPath,'simulation_time',simTime);
% Persist one-row result for aggregator
row=struct('seed',seed,'policy',char(policy), ...
    'n_success',gf(summary,'n_success',0),'n_workflows',gf(summary,'n_workflows',0), ...
    'n_expired',gf(summary,'n_expired',0),'n_remote_stages_done',gf(summary,'n_remote_stages_done',0), ...
    'radio_data_fragments',gf(summary,'radio_data_fragments',0), ...
    'cpu_served',gf(summary,'cpu_served',0),'ok',true);
writetable(struct2table(row),fullfile(caseRoot,'r3m_row.csv'));
fprintf('R3-M shard done seed=%d policy=%s n_success=%g\n',seed,policy,row.n_success);
end

function v=gf(s,f,d)
if isstruct(s) && isfield(s,f), v=s.(f); else, v=d; end
if isstring(v)||ischar(v), v=str2double(v); if isnan(v), v=d; end, end
end
