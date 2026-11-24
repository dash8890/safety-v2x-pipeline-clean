function out=runR3MGate()
%RUNR3MGATE One-time model-validity gate (predeclared r3m_gate.json).
% No retuning. Policies: B0, B2, B3, B4, B4-immediate release ablation.
root=fileparts(mfilename('fullpath'));
addpath(genpath(root));
cfgPath=fullfile(root,'config','r3m_gate.json');
cfg=jsondecode(fileread(cfgPath));

stamp=string(datetime('now','Format','yyyyMMdd_HHmmss'));
outRoot=fullfile(root,'outputs',sprintf('r3m_gate_%s',stamp));
mkdir(outRoot);
copyfile(cfgPath,fullfile(outRoot,'r3m_gate_FROZEN.json'));

seeds=double(cfg.experiment.seeds(:))';
policies=string(cfg.experiment.policies(:))';
simTime=double(cfg.experiment.duration_s);

rows=struct('seed',{},'policy',{},'n_success',{},'n_workflows',{},'n_expired',{}, ...
    'n_remote_stages_done',{},'radio_data_fragments',{},'cpu_served',{},'ok',{});
k=0;
for seed=seeds
    for pol=policies
        k=k+1;
        caseRoot=fullfile(outRoot,sprintf('seed%d_%s',seed,pol));
        fprintf('R3-M seed=%d policy=%s\n',seed,pol);
        try
            s=runR3PilotCase(caseRoot,pol,seed,'config',cfgPath,'simulation_time',simTime);
            rows(k)=struct('seed',seed,'policy',pol, ...
                'n_success',gf(s,'n_success',0),'n_workflows',gf(s,'n_workflows',0), ...
                'n_expired',gf(s,'n_expired',0),'n_remote_stages_done',gf(s,'n_remote_stages_done',0), ...
                'radio_data_fragments',gf(s,'radio_data_fragments',0), ...
                'cpu_served',gf(s,'cpu_served',0),'ok',true);
        catch ME
            rows(k)=struct('seed',seed,'policy',pol,'n_success',0,'n_workflows',0, ...
                'n_expired',0,'n_remote_stages_done',0,'radio_data_fragments',0, ...
                'cpu_served',0,'ok',false);
            fprintf('  FAIL: %s\n',ME.message);
            if ~isfolder(caseRoot), mkdir(caseRoot); end
            fid=fopen(fullfile(caseRoot,'ERROR.txt'),'w');
            if fid>0, fprintf(fid,'%s',getReport(ME)); fclose(fid); end
        end
    end
end

T=struct2table(rows);
writetable(T,fullfile(outRoot,'r3m_results.csv'));
fid=fopen(fullfile(outRoot,'R3M_FREEZE.txt'),'w');
fprintf(fid,'R3-M frozen from config/r3m_gate.json before first run.\n');
fprintf(fid,'Arrivals: 12 workflows @ 30ms spacing from 0.10s.\n');
fprintf(fid,'Contact: helper-2 exit 0.22s / return 0.40s (scripted, labeled).\n');
fprintf(fid,'Deadline 500ms. No post-hoc retuning.\n');
fclose(fid);
out=struct('outRoot',outRoot,'table',T,'rows',rows);
fprintf('R3-M complete: %s\n',outRoot);
disp(T);
end

function v=gf(s,f,d)
if isstruct(s) && isfield(s,f), v=s.(f); else, v=d; end
if isstring(v)||ischar(v), v=str2double(v); if isnan(v), v=d; end, end
end
