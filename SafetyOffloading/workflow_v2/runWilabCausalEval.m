function runWilabCausalEval(varargin)
%RUNWILABCAUSALEVAL Matched WiLab Mode-2 holdout.
% Principal baseline: ValueGreedy.
% Learning contribution: PDP teacher vs BC-only vs Full Causal DDQN (PDP ≠ literature baseline).
% runWilabCausalEval
% runWilabCausalEval('seeds',[300 301],'policies',["workflow_value_greedy","workflow_causal_ddqn"])
seeds=300:309;
policies=["workflow_value_greedy","workflow_drop_low","workflow_causal_ddqn"];
for i=1:2:numel(varargin)
    switch string(varargin{i})
        case "seeds", seeds=double(varargin{i+1}(:)');
        case "policies", policies=string(varargin{i+1}(:)');
    end
end
root=fileparts(mfilename('fullpath'));
cfgPath=fullfile(root,'config','wilab_causal_eval.json');
stamp=string(datetime('now','Format','yyyyMMdd_HHmmss'));
outRoot=fullfile(root,'outputs',sprintf('wilab_causal_%s',stamp));
if ~isfolder(outRoot), mkdir(outRoot); end
copyfile(cfgPath,fullfile(outRoot,'config_base.json'));

rows={};
for p=1:numel(policies)
    for s=1:numel(seeds)
        pol=policies(p); seed=seeds(s);
        caseRoot=fullfile(outRoot,sprintf('seed%d_%s',seed,pol));
        fprintf('WiLab causal %s seed=%d -> %s\n',pol,seed,caseRoot);
        try
            summary=runR3PilotCase(caseRoot,pol,seed,'config',cfgPath,'radio_mode','wilab_mode2');
        catch ME
            summary=struct('policy',char(pol),'seed',seed,'n_success',0,'n_workflows',0, ...
                'total_value',0,'error',ME.message);
            warning('run failed: %s',ME.message);
        end
        tv=0; if isfield(summary,'total_value'), tv=double(summary.total_value); end
        ns=0; if isfield(summary,'n_success'), ns=double(summary.n_success); end
        rows(end+1,:)={char(pol),seed,tv,ns}; %#ok<AGROW>
        fprintf('  V=%.1f N=%.0f\n',tv,ns);
    end
end
T=cell2table(rows,'VariableNames',{'policy','seed','total_value','n_success'});
writetable(T,fullfile(outRoot,'per_seed.csv'));

% Aggregate vs ValueGreedy
agg=struct();
for p=1:numel(policies)
    pol=policies(p);
    mask=strcmp(T.policy,char(pol));
    v=T.total_value(mask); n=T.n_success(mask);
    agg.(matlab.lang.makeValidName(char(pol)))=struct( ...
        'mean_value',mean(v),'std_value',std(v), ...
        'mean_success',mean(n),'std_success',std(n), ...
        'per_seed_value',v(:)','per_seed_success',n(:)');
end
vgName=matlab.lang.makeValidName('workflow_value_greedy');
ddName=matlab.lang.makeValidName('workflow_causal_ddqn');
result=struct('out_root',char(outRoot),'seeds',seeds,'policies',cellstr(policies),'agg',agg);
if isfield(agg,vgName) && isfield(agg,ddName)
    vg=agg.(vgName).mean_value; dd=agg.(ddName).mean_value;
    result.delta_value=dd-vg;
    result.relative_value_pct=100*(dd-vg)/max(vg,eps);
    result.utility_superior_vs_ValueGreedy=result.relative_value_pct>=10 && result.delta_value>=2;
end
fid=fopen(fullfile(outRoot,'wilab_causal_holdout.json'),'w');
fprintf(fid,'%s',jsonencode(result,'PrettyPrint',true)); fclose(fid);
fprintf('WROTE %s\n',fullfile(outRoot,'wilab_causal_holdout.json'));
if isfield(result,'relative_value_pct')
    fprintf('DDQN vs VG: dV=%.2f (%.1f%%) superior=%d\n', ...
        result.delta_value,result.relative_value_pct,result.utility_superior_vs_ValueGreedy);
end
end
