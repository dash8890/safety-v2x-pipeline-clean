function out=aggregateR3M(outRoot)
%AGGREGATER3M Collect summary.json rows from an R3-M output root.
root=fileparts(mfilename('fullpath'));
addpath(genpath(root));
d=dir(fullfile(outRoot,'seed*'));
rows=struct('seed',{},'policy',{},'n_success',{},'n_workflows',{},'n_expired',{}, ...
    'n_remote_stages_done',{},'radio_data_fragments',{},'cpu_served',{},'ok',{});
k=0;
for i=1:numel(d)
    sp=fullfile(d(i).folder,d(i).name,'workflow_v2','summary.json');
    if ~isfile(sp), continue; end
    s=jsondecode(fileread(sp));
    tok=regexp(d(i).name,'^seed(\d+)_(.+)$','tokens','once');
    if isempty(tok), continue; end
    k=k+1;
    rows(k)=struct('seed',str2double(tok{1}),'policy',string(tok{2}), ...
        'n_success',gf(s,'n_success',0),'n_workflows',gf(s,'n_workflows',0), ...
        'n_expired',gf(s,'n_expired',0),'n_remote_stages_done',gf(s,'n_remote_stages_done',0), ...
        'radio_data_fragments',gf(s,'radio_data_fragments',0), ...
        'cpu_served',gf(s,'cpu_served',0),'ok',true);
end
T=struct2table(rows);
if height(T)>0
    T=sortrows(T,{'seed','policy'});
end
writetable(T,fullfile(outRoot,'r3m_results.csv'));
out=struct('outRoot',outRoot,'table',T,'rows',rows);
disp(T);
end

function v=gf(s,f,d)
if isstruct(s) && isfield(s,f), v=s.(f); else, v=d; end
if isstring(v)||ischar(v), v=str2double(v); if isnan(v), v=d; end, end
end
