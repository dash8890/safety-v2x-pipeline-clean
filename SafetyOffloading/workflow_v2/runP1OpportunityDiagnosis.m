function out=runP1OpportunityDiagnosis(outRoot,varargin)
%RUNP1OPPORTUNITYDIAGNOSIS Fixed development-only physical headroom matrix.
% Six predeclared cells: low/moderate/high real CPU contention x favourable /
% intermittent contact, five matched WiLab seeds each.  It is an opportunity
% diagnosis, never a final comparison or a DRL selection result.
root=fileparts(mfilename('fullpath')); cfgBase=jsondecode(fileread(fullfile(root,'config','p1_opportunity.json')));
if nargin<1 || strlength(string(outRoot))==0
    outRoot=fullfile(root,'outputs','p1_opportunity_active');
end
if ~isfolder(outRoot), mkdir(outRoot); end
seeds=double(cfgBase.experiment.development_seeds);
loads=[0.30 0.65 0.85]; contacts=["favorable" "intermittent"];
policies=["workflow_b0_local" "workflow_b4_joint"];
rows={};
for li=1:numel(loads)
    for ci=1:numel(contacts)
        cfg=cfgBase; cfg.workflow.cpu_offered_load=loads(li); cfg.workflow.force_scripted_contact=true;
        if contacts(ci)=="favorable"
            cfg.workflow.contact_schedule=struct('t_s',0,'links',{{[1 2 1;1 3 1;2 3 1]}});
        else
            cfg.workflow.contact_schedule=[ ...
                struct('t_s',0,'links',{{[1 2 1;1 3 1;2 3 1]}}), ...
                struct('t_s',0.30,'links',{{[1 2 0;1 3 1;2 3 1]}}), ...
                struct('t_s',0.70,'links',{{[1 2 1;1 3 1;2 3 1]}})];
        end
        cellId=sprintf('L%d_%s',li,contacts(ci)); cellRoot=fullfile(outRoot,cellId); if ~isfolder(cellRoot), mkdir(cellRoot); end
        cfgPath=fullfile(cellRoot,'config.json'); fid=fopen(cfgPath,'w'); fprintf(fid,'%s',jsonencode(cfg,'PrettyPrint',true)); fclose(fid);
        for si=1:numel(seeds)
            for pi=1:numel(policies)
                pol=policies(pi); caseRoot=fullfile(cellRoot,sprintf('seed%d_%s',seeds(si),pol));
                S=runR3PilotCase(caseRoot,pol,seeds(si),'config',cfgPath,'simulation_time',cfg.experiment.duration_s);
                rows(end+1,:)={cellId,loads(li),contacts(ci),seeds(si),pol,field(S,'total_value',0),field(S,'n_success',0),field(S,'n_remote_stages_done',0),field(S,'radio_data_fragments',0)}; %#ok<AGROW>
            end
        end
    end
end
T=cell2table(rows,'VariableNames',{'cell','cpu_load','contact','seed','policy','utility','success','remote_stages','radio_fragments'});
writetable(T,fullfile(outRoot,'per_seed.csv'));
save(fullfile(outRoot,'P1_COMPLETE.mat'),'T','cfgBase','seeds'); out=T;
end

function v=field(S,name,default)
if isfield(S,name), v=double(S.(name)); else, v=default; end
end
