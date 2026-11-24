function out=runWorkflowPilot(varargin)
%RUNWORKFLOWPILOT R3 three-seed joint-decision relevance pilot (frozen params).
% out = runWorkflowPilot()
% out = runWorkflowPilot('smoke')          % one seed, B0+B4, Mode-2 only
% out = runWorkflowPilot('pilot')          % seeds 1..3, B0-B4 + abstract twin
root=fileparts(mfilename('fullpath'));
addpath(genpath(root));
mode="pilot";
if nargin>=1 && ~isempty(varargin{1}), mode=string(varargin{1}); end

stamp=string(datetime('now','Format','yyyyMMdd_HHmmss'));
outRoot=fullfile(root,'outputs',sprintf('r3_%s_%s',mode,stamp));
if ~isfolder(outRoot), mkdir(outRoot); end

if mode=="smoke"
    seeds=1;
    policies=["workflow_b0_local","workflow_b4_joint"];
    doAbstract=false;
    doB5=false;
else
    seeds=[1 2 3];
    policies=["workflow_b0_local","workflow_b1_greedy","workflow_b2_queue", ...
        "workflow_b3_mobility","workflow_b4_joint"];
    doAbstract=true;
    doB5=true;
end

rows=struct('seed',{},'policy',{},'radio_mode',{},'n_success',{},'n_workflows',{}, ...
    'n_expired',{},'cpu_served',{},'radio_data_fragments',{},'ok',{});
k=0;
for seed=seeds
    for pol=policies
        k=k+1;
        caseRoot=fullfile(outRoot,sprintf('seed%d_%s_wilab',seed,pol));
        fprintf('R3 %s seed=%d policy=%s radio=wilab_mode2\n',mode,seed,pol);
        try
            s=runR3PilotCase(caseRoot,pol,seed,'radio_mode','wilab_mode2');
            rows(k)=rowFrom(s,seed,pol,"wilab_mode2",true);
        catch ME
            rows(k)=struct('seed',seed,'policy',pol,'radio_mode',"wilab_mode2", ...
                'n_success',0,'n_workflows',0,'n_expired',0,'cpu_served',0, ...
                'radio_data_fragments',0,'ok',false);
            fprintf('  FAIL: %s\n',ME.message);
            fid=fopen(fullfile(caseRoot,'ERROR.txt'),'w'); if fid>0, fprintf(fid,'%s',getReport(ME)); fclose(fid); end
        end
    end
    if doAbstract
        k=k+1;
        pol="workflow_b4_joint";
        caseRoot=fullfile(outRoot,sprintf('seed%d_%s_abstract',seed,pol));
        fprintf('R3 %s seed=%d policy=%s radio=abstract_link\n',mode,seed,pol);
        try
            s=runR3PilotCase(caseRoot,pol,seed,'radio_mode','abstract_link');
            rows(k)=rowFrom(s,seed,pol,"abstract_link",true);
        catch ME
            rows(k)=struct('seed',seed,'policy',pol,'radio_mode',"abstract_link", ...
                'n_success',0,'n_workflows',0,'n_expired',0,'cpu_served',0, ...
                'radio_data_fragments',0,'ok',false);
            fprintf('  FAIL: %s\n',ME.message);
        end
    end
    if doB5 && seed==seeds(1)
        k=k+1;
        pol="workflow_b5_privileged";
        caseRoot=fullfile(outRoot,sprintf('seed%d_%s_wilab',seed,pol));
        fprintf('R3 %s seed=%d policy=%s (diagnostic)\n',mode,seed,pol);
        try
            s=runR3PilotCase(caseRoot,pol,seed,'radio_mode','wilab_mode2');
            rows(k)=rowFrom(s,seed,pol,"wilab_mode2",true);
        catch ME
            rows(k)=struct('seed',seed,'policy',pol,'radio_mode',"wilab_mode2", ...
                'n_success',0,'n_workflows',0,'n_expired',0,'cpu_served',0, ...
                'radio_data_fragments',0,'ok',false);
            fprintf('  FAIL: %s\n',ME.message);
        end
    end
end

T=struct2table(rows);
writetable(T,fullfile(outRoot,'pilot_results.csv'));
out=struct('outRoot',outRoot,'rows',rows,'table',T);
% Freeze note
fid=fopen(fullfile(outRoot,'R3_PILOT_FREEZE.txt'),'w');
fprintf(fid,'R3 pilot frozen params: diamond4, 36 Mcycles, 500ms, load=0.80, arrival=[0.10]s, sim=1.0s, helpers=2\n');
fprintf(fid,'No post-hoc workload retuning. One corrective iteration for correctness only.\n');
fclose(fid);
fprintf('R3 %s complete: %s\n',mode,outRoot);
end

function r=rowFrom(s,seed,pol,rmode,ok)
r=struct('seed',seed,'policy',string(pol),'radio_mode',string(rmode), ...
    'n_success',getf(s,'n_success',0),'n_workflows',getf(s,'n_workflows',0), ...
    'n_expired',getf(s,'n_expired',0),'cpu_served',getf(s,'cpu_served',0), ...
    'radio_data_fragments',getf(s,'radio_data_fragments',0),'ok',ok);
end

function v=getf(s,f,d)
if isstruct(s) && isfield(s,f), v=s.(f); else, v=d; end
if isstring(v)||ischar(v), v=str2double(v); if isnan(v), v=d; end, end
end
