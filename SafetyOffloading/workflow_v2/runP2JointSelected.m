function T=runP2JointSelected(outRoot)
%RUNP2JOINTSELECTED Matched five-seed B4 comparison in P1's selected regime.
root=fileparts(mfilename('fullpath'));
cfg0=jsondecode(fileread(fullfile(root,'config','p1_opportunity.json')));
if nargin<1 || strlength(string(outRoot))==0
    outRoot=fullfile(root,'outputs','p2_joint_selected');
end
if ~isfolder(outRoot), mkdir(outRoot); end
rows={}; seeds=double(cfg0.experiment.development_seeds);
for contact=["favorable" "intermittent"]
    cfg=cfg0; cfg.workflow.cpu_offered_load=0.85;
    cfg.workflow.force_scripted_contact=true;
    if contact=="favorable"
        cfg.workflow.contact_schedule=struct('t_s',0,'links',{{[1 2 1;1 3 1;2 3 1]}});
    else
        cfg.workflow.contact_schedule=[struct('t_s',0,'links',{{[1 2 1;1 3 1;2 3 1]}}), ...
          struct('t_s',0.30,'links',{{[1 2 0;1 3 1;2 3 1]}}), ...
          struct('t_s',0.70,'links',{{[1 2 1;1 3 1;2 3 1]}})];
    end
    tag="L3_"+contact; dir=fullfile(outRoot,tag); if ~isfolder(dir), mkdir(dir); end
    cfgPath=fullfile(dir,'config.json'); fid=fopen(cfgPath,'w'); fprintf(fid,'%s',jsonencode(cfg,'PrettyPrint',true)); fclose(fid);
    for seed=seeds'
        S=runR3PilotCase(fullfile(dir,sprintf('seed%d_workflow_b4_joint',seed)), ...
          'workflow_b4_joint',seed,'config',cfgPath,'simulation_time',cfg.experiment.duration_s);
        rows(end+1,:)={tag,contact,seed,S.total_value,S.n_success,S.n_remote_stages_done,S.radio_data_fragments}; %#ok<AGROW>
    end
end
T=cell2table(rows,'VariableNames',{'cell','contact','seed','utility','success','remote_stages','radio_fragments'});
writetable(T,fullfile(outRoot,'per_seed.csv')); save(fullfile(outRoot,'P2_COMPLETE.mat'),'T');
end
