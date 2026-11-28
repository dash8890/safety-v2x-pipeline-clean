function manifest=buildP7Datasets(varargin)
%BUILDP7DATASETS Assemble P7 prediction/experience tables from finished runs.
% Does not launch simulations. Reads P4/P6 case outputs and writes:
%   - seed-partitioned prediction tables (train vs validation never mixed)
%   - action-level conditional risk rows (never filled with policy harm)
%   - policy-level paired safety costs (separate table)
%   - coverage_audit.json + deficiencies.json before any extra data gen
p=inputParser;
addParameter(p,'ConfigPath',fullfile(fileparts(mfilename('fullpath')),'config','p7_data_generation.json'));
addParameter(p,'OutputRoot',"");
parse(p,varargin{:});
root=fileparts(mfilename('fullpath')); addpath(genpath(root));
cfg=jsondecode(fileread(p.Results.ConfigPath));
outRoot=p.Results.OutputRoot;
if strlength(string(outRoot))==0
    outRoot=fullfile(root,'outputs','p7_datasets');
end
if ~isfolder(outRoot), mkdir(outRoot); end

p4=fullfile(root,cfg.sources.p4_baseline_root);
p6=fullfile(root,cfg.sources.p6_myopic_root);
configs=unique([cfg.sources.primary_configs(:); cfg.sources.negative_control_configs(:); cfg.sources.generalization_configs(:)]);
seeds=unique([cfg.sources.train_seeds(:); cfg.sources.validation_seeds(:)]);
heldout=cfg.sources.heldout_seeds_reserved_for_p10(:);
assertPartitionDisjoint(cfg.sources.train_seeds,cfg.sources.validation_seeds,heldout);

completion=table(); latency=table(); actionRisk=table();
policySafety=table(); experience=table();
sourceInventory=struct('configuration_id',{},'seed',{},'policy',{},'folder',{}, ...
    'has_tasks',{},'has_assignment_features',{},'has_schedule_epochs',{},'has_safety_packets',{});

for ci=1:numel(configs)
    for si=1:numel(seeds)
        cfgId=configs(ci); seed=seeds(si);
        if ismember(seed,heldout)
            error('P7 refused held-out seed %d reserved for P10.',seed);
        end
        caseDir=fullfile(p4,sprintf('cfg%02d_seed%d',cfgId,seed));
        if ~isfolder(caseDir), continue; end
        worlds=discoverWorlds(caseDir,p6,cfgId,seed);
        baseDir=worlds.deadline_local;
        for wi=1:numel(worlds.policies)
            pol=worlds.policies(wi);
            inv=struct('configuration_id',cfgId,'seed',seed,'policy',string(pol.name), ...
                'folder',string(pol.folder), ...
                'has_tasks',isfile(fullfile(pol.folder,'tasks.csv')), ...
                'has_assignment_features',isfile(fullfile(pol.folder,'assignment_features.csv')), ...
                'has_schedule_epochs',isfile(fullfile(pol.folder,'schedule_epochs.csv')), ...
                'has_safety_packets',isfile(fullfile(pol.folder,'safety_packets.csv')));
            sourceInventory(end+1)=inv; %#ok<AGROW>
            [cTab,lTab,aTab,sTab,eTab]=extractPolicy(pol.folder,cfgId,seed,pol.name,baseDir);
            completion=[completion; cTab]; %#ok<AGROW>
            latency=[latency; lTab]; %#ok<AGROW>
            actionRisk=[actionRisk; aTab]; %#ok<AGROW>
            policySafety=[policySafety; sTab]; %#ok<AGROW>
            experience=[experience; eTab]; %#ok<AGROW>
        end
    end
end

trainSeeds=cfg.sources.train_seeds(:)';
valSeeds=cfg.sources.validation_seeds(:);
completion=tagPartition(completion,trainSeeds,valSeeds);
latency=tagPartition(latency,trainSeeds,valSeeds);
actionRisk=tagPartition(actionRisk,trainSeeds,valSeeds);
policySafety=tagPartition(policySafety,trainSeeds,valSeeds);
experience=tagPartition(experience,trainSeeds,valSeeds);

writetable(completion,fullfile(outRoot,'completion_prediction.csv'));
writetable(latency,fullfile(outRoot,'latency_prediction.csv'));
writetable(actionRisk,fullfile(outRoot,'action_conditional_risk.csv'));
writetable(policySafety,fullfile(outRoot,'policy_safety_costs.csv'));
% Keep legacy filename as an alias pointing at policy-level rows only.
writetable(policySafety,fullfile(outRoot,'safety_prediction.csv'));
writetable(experience,fullfile(outRoot,'sequential_experience.csv'));

audit=auditCoverage(completion,actionRisk,experience,sourceInventory,cfg,p4,p6,configs,seeds);
deficiencies=collectDeficiencies(audit,cfg);
fid=fopen(fullfile(outRoot,'coverage_audit.json'),'wt');
fprintf(fid,'%s',jsonencode(audit,'PrettyPrint',true)); fclose(fid);
fid=fopen(fullfile(outRoot,'deficiencies.json'),'wt');
fprintf(fid,'%s',jsonencode(deficiencies,'PrettyPrint',true)); fclose(fid);

manifest=struct( ...
    'output_root',string(outRoot), ...
    'completion_rows',height(completion), ...
    'latency_rows',height(latency), ...
    'action_conditional_risk_rows',height(actionRisk), ...
    'policy_safety_rows',height(policySafety), ...
    'experience_rows',height(experience), ...
    'configs',configs(:)','seeds',seeds(:)', ...
    'train_seeds',trainSeeds(:)','validation_seeds',valSeeds(:)', ...
    'partition_rule',"scenario_seed_strict", ...
    'safety_label_rule',"action_conditional_risk_separate_from_policy_costs", ...
    'deficiency_count',numel(deficiencies.items), ...
    'ready_for_calibration',deficiencies.ready_for_calibration, ...
    'needs_additional_experience',deficiencies.needs_additional_experience);
fid=fopen(fullfile(outRoot,'manifest.json'),'wt');
fprintf(fid,'%s',jsonencode(manifest,'PrettyPrint',true)); fclose(fid);
fprintf(['P7 datasets: completion=%d latency=%d action_risk=%d policy_safety=%d ' ...
    'experience=%d deficiencies=%d ready_for_calibration=%d -> %s\n'], ...
    height(completion),height(latency),height(actionRisk),height(policySafety), ...
    height(experience),numel(deficiencies.items),deficiencies.ready_for_calibration,outRoot);
end

function assertPartitionDisjoint(trainSeeds,valSeeds,heldout)
overlapTV=intersect(trainSeeds(:),valSeeds(:));
if ~isempty(overlapTV)
    error('Train/validation seeds overlap: %s',mat2str(overlapTV(:)'));
end
overlapTH=intersect(trainSeeds(:),heldout(:));
overlapVH=intersect(valSeeds(:),heldout(:));
if ~isempty(overlapTH) || ~isempty(overlapVH)
    error('Train/validation seeds collide with P10 held-out seeds.');
end
end

function worlds=discoverWorlds(caseDir,p6Root,cfgId,seed)
worlds.deadline_local=fullfile(caseDir,'deadline_local','safety_offloading');
worlds.policies=struct('name',{},'folder',{});
candNames=["radio_search","safety_search","risk_myopic","ideal_oracle"];
summaryFile=fullfile(caseDir,'case_summary.csv');
if isfile(summaryFile)
    tab=readtable(summaryFile,'VariableNamingRule','preserve');
    if ismember('source_world',tab.Properties.VariableNames)
        for i=1:height(tab)
            src=string(tab.source_world(i));
            if src=="deadline_local" || src=="", continue; end
            folder=fullfile(caseDir,src,'safety_offloading');
            if isfolder(folder)
                worlds.policies(end+1)=struct('name',string(tab.method(i)),'folder',folder); %#ok<AGROW>
            end
        end
    end
end
if isempty(worlds.policies)
    for n=candNames
        folder=fullfile(caseDir,n,'safety_offloading');
        if isfolder(folder)
            worlds.policies(end+1)=struct('name',n,'folder',folder); %#ok<AGROW>
        end
    end
else
    keep=true(1,numel(worlds.policies));
    for i=1:numel(worlds.policies)
        keep(i)=~startsWith(string(worlds.policies(i).name),"safety_cand");
    end
    worlds.policies=worlds.policies(keep);
end
% Include deadline_local for local-execution coverage / completion labels.
if isfolder(worlds.deadline_local)
    worlds.policies=[struct('name',"deadline_local",'folder',worlds.deadline_local), worlds.policies];
end
myopic=fullfile(p6Root,sprintf('cfg%02d_seed%d',cfgId,seed),'risk_myopic_p6','safety_offloading');
if isfolder(myopic)
    worlds.policies(end+1)=struct('name',"risk_myopic_p6",'folder',myopic);
end
end

function [completion,latency,actionRisk,policySafety,experience]=extractPolicy(so,cfgId,seed,policyName,baseDir)
completion=table(); latency=table(); actionRisk=table(); policySafety=table(); experience=table();
tasksFile=fullfile(so,'tasks.csv');
if ~isfile(tasksFile), return; end
tasks=readtable(tasksFile,'VariableNamingRule','preserve');
features=table();
featFile=fullfile(so,'assignment_features.csv');
if isfile(featFile), features=readtable(featFile,'VariableNamingRule','preserve'); end
decisions=table();
decFile=fullfile(so,'decisions.csv');
if isfile(decFile), decisions=readtable(decFile,'VariableNamingRule','preserve'); end

% --- Completion: executed actions (realized labels) ---
for i=1:height(tasks)
    tid=tasks.task_id(i);
    proc=tasks.destination(i);
    if ~isfinite(proc) || proc==0, proc=tasks.origin_id(i); end
    act=string(tasks.action(i));
    isOff=act=="offload";
    isLocal=act=="local";
    feas="NA";
    if ~isempty(decisions) && ismember('feasibility_class',decisions.Properties.VariableNames)
        hitD=decisions.task_id==tid;
        if any(hitD), feas=string(decisions.feasibility_class(find(hitD,1))); end
    end
    % completed_on_time stored as double so counterfactual NaNs can share the column.
    row=table(cfgId,seed,string(policyName),tid,proc,act,logical(isOff),logical(isLocal), ...
        double(tasks.completed_on_time(i)),double(tasks.cycles(i)),double(tasks.in_bytes(i)), ...
        double(tasks.out_bytes(i)),double(tasks.deadline_ms(i)-tasks.release_ms(i)),feas,true, ...
        'VariableNames',{'configuration_id','seed','policy','task_id','processor_id','action','is_offload', ...
        'is_local_action','completed_on_time','cycles','in_bytes','out_bytes','deadline_horizon_ms', ...
        'feasibility_class','is_realized'});
    if ~isempty(features)
        hit=features.task_id==tid & features.processor_id==proc & features.chosen==1;
        if ~any(hit), hit=features.task_id==tid & features.chosen==1; end
        if any(hit)
            f=features(find(hit,1),:);
            row=attachFeatureColumns(row,f);
        end
    end
    completion=[completion; row]; %#ok<AGROW>
end

% Counterfactual candidates from assignment_features (coverage only; no realized label reuse).
if ~isempty(features)
    for i=1:height(features)
        if ismember('chosen',features.Properties.VariableNames) && features.chosen(i)==1
            continue
        end
        f=features(i,:);
        act="candidate";
        if ismember('is_local',f.Properties.VariableNames) && logical(f.is_local)
            act="candidate_local";
        else
            act="candidate_offload";
        end
        feas="NA";
        if ismember('feasibility_class',f.Properties.VariableNames)
            feas=string(f.feasibility_class);
        end
        row=table(cfgId,seed,string(policyName),double(f.task_id),double(f.processor_id),act, ...
            act=="candidate_offload",act=="candidate_local",NaN,NaN,NaN,NaN,NaN,feas,false, ...
            'VariableNames',{'configuration_id','seed','policy','task_id','processor_id','action','is_offload', ...
            'is_local_action','completed_on_time','cycles','in_bytes','out_bytes','deadline_horizon_ms', ...
            'feasibility_class','is_realized'});
        row=attachFeatureColumns(row,f);
        completion=[completion; row]; %#ok<AGROW>
    end
end

% --- Latency: successful timing paths for offloads ---
off=tasks(string(tasks.action)=="offload",:);
for i=1:height(off)
    inLat=off.input_delivered_ms(i)-off.release_ms(i);
    comp=off.cpu_finish_ms(i)-off.cpu_start_ms(i);
    outLat=off.result_delivered_ms(i)-off.cpu_finish_ms(i);
    e2e=off.result_delivered_ms(i)-off.release_ms(i);
    q=off.cpu_start_ms(i)-off.input_delivered_ms(i);
    latency=[latency; {cfgId,seed,string(policyName),off.task_id(i),off.destination(i), ...
        logical(off.completed_on_time(i)),inLat,q,comp,outLat,e2e}]; %#ok<AGROW>
end
if ~isempty(latency)
    latency.Properties.VariableNames={'configuration_id','seed','policy','task_id','processor_id', ...
        'completed_on_time','input_latency_ms','helper_queue_ms','compute_ms','result_latency_ms','e2e_ms'};
end

% --- Action-level conditional risk (NOT policy harm) ---
if ~isempty(features)
    for i=1:height(features)
        f=features(i,:);
        chosen=false;
        if ismember('chosen',f.Properties.VariableNames), chosen=logical(f.chosen); end
        actionRisk=[actionRisk; {cfgId,seed,string(policyName),f.decision_id,f.task_id,f.processor_id, ...
            logical(f.is_local),chosen,string(f.feasibility_class), ...
            f.predicted_conditional_safety_risk,f.predicted_completion_probability, ...
            f.relative_distance_m,f.masked,"action_level_provisional"}]; %#ok<AGROW>
    end
elseif ~isempty(decisions)
    % P4 worlds: provisional risk_upper on the chosen decision only.
    for i=1:height(decisions)
        risk=NaN;
        if ismember('risk_upper',decisions.Properties.VariableNames)
            risk=decisions.risk_upper(i);
        end
        feas="NA";
        if ismember('feasibility_class',decisions.Properties.VariableNames)
            feas=string(decisions.feasibility_class(i));
        end
        isLocal=string(decisions.chosen_action(i))=="local";
        actionRisk=[actionRisk; {cfgId,seed,string(policyName),decisions.decision_id(i), ...
            decisions.task_id(i),decisions.helper_id(i),isLocal,true,feas,risk,NaN, ...
            decisions.helper_distance_m(i),false,"action_level_provisional_p4"}]; %#ok<AGROW>
    end
end
if ~isempty(actionRisk)
    actionRisk.Properties.VariableNames={'configuration_id','seed','policy','decision_id','task_id', ...
        'processor_id','is_local','chosen','feasibility_class','action_conditional_risk', ...
        'predicted_completion_probability','relative_distance_m','masked','label_kind'};
end

% --- Policy-level paired safety (never join onto individual actions) ---
polSafety=fullfile(so,'safety_packets.csv');
if isfolder(baseDir) && isfile(fullfile(baseDir,'safety_packets.csv')) && isfile(polSafety)
    B=readtable(fullfile(baseDir,'safety_packets.csv'),'VariableNamingRule','preserve');
    P=readtable(polSafety,'VariableNamingRule','preserve');
    try
        [overall,~,paired]=summarizePolicySafety(B,P,0.01,0.10);
        policySafety=table(repmat(cfgId,height(paired),1),repmat(seed,height(paired),1), ...
            repmat(string(policyName),height(paired),1), ...
            paired.flow_id,paired.tx_id,paired.rx_id,paired.distance_bin, ...
            paired.baseline_success,paired.policy_success,paired.harmful_flip,paired.avoided_failure, ...
            'VariableNames',{'configuration_id','seed','policy','flow_id','tx_id','rx_id','distance_bin', ...
            'baseline_success','policy_success','harmful_flip','avoided_failure'});
        policySafety.policy_failure=~policySafety.policy_success;
        policySafety.label_kind=repmat("policy_level_paired",height(policySafety),1);
        policySafety.case_harm_cost=repmat(overall.harmful_flips-0.01*overall.opportunities,height(policySafety),1);
        policySafety.case_failure_cost=repmat( ...
            overall.opportunities*overall.absolute_failure_probability-0.10*overall.opportunities, ...
            height(policySafety),1);
    catch
        policySafety=table();
    end
end

% --- Sequential experience: per-decision transitions; no aggregate harm broadcast ---
epFile=fullfile(so,'schedule_epochs.csv');
if isfile(epFile)
    ep=readtable(epFile,'VariableNamingRule','preserve');
    ep=sortrows(ep,{'epoch_ms','decision_id'});
    for i=1:height(ep)
        nextId=NaN; done=i==height(ep);
        if i<height(ep), nextId=ep.decision_id(i+1); end
        experience=[experience; {cfgId,seed,string(policyName),ep.decision_id(i),ep.epoch_ms(i), ...
            string(ep.action_type(i)),ep.task_id(i),ep.processor_id(i), ...
            ep.pending_tasks(i),ep.candidate_assignments(i),nextId,done, ...
            NaN,NaN,"per_decision_no_policy_harm"}]; %#ok<AGROW>
    end
elseif isfile(decFile)
    decisions=sortrows(decisions,{'time_ms','decision_id'});
    for i=1:height(decisions)
        nextId=NaN; done=i==height(decisions);
        if i<height(decisions), nextId=decisions.decision_id(i+1); end
        experience=[experience; {cfgId,seed,string(policyName),decisions.decision_id(i),decisions.time_ms(i), ...
            string(decisions.chosen_action(i)),decisions.task_id(i),decisions.helper_id(i), ...
            NaN,NaN,nextId,done,NaN,NaN,"legacy_decisions_chain"}]; %#ok<AGROW>
    end
end
if ~isempty(experience)
    experience.Properties.VariableNames={'configuration_id','seed','policy','decision_id','time_ms', ...
        'action_type','task_id','processor_id','pending_tasks','candidate_assignments', ...
        'next_decision_id','done','reward','action_level_harm_cost','label_kind'};
    rewFile=fullfile(so,'rewards.csv');
    if isfile(rewFile)
        rew=readtable(rewFile,'VariableNamingRule','preserve');
        if ismember('task_id',rew.Properties.VariableNames) && ismember('reward',rew.Properties.VariableNames)
            for i=1:height(experience)
                hit=rew.task_id==experience.task_id(i);
                if any(hit), experience.reward(i)=sum(rew.reward(hit)); end
            end
        end
    end
    % Intentionally do NOT copy safety_costs.csv totals onto each action.
end
end

function row=attachFeatureColumns(row,f)
cols={ ...
    'predicted_completion_probability','predicted_deadline_slack_ms', ...
    'predicted_conditional_safety_risk','processor_cpu_ghz','is_local', ...
    'estimated_input_latency_ms','estimated_helper_queueing_ms', ...
    'estimated_computation_latency_ms','estimated_result_latency_ms','relative_distance_m'};
for k=1:numel(cols)
    c=cols{k};
    if ismember(c,f.Properties.VariableNames)
        row.(c)=f.(c);
    end
end
if ismember('feasibility_class',f.Properties.VariableNames) && ...
        (~ismember('feasibility_class',row.Properties.VariableNames) || row.feasibility_class=="NA")
    row.feasibility_class=string(f.feasibility_class);
end
end

function t=tagPartition(t,trainSeeds,valSeeds)
if isempty(t)
    t=table(); t.partition=strings(0,1); return
end
part=strings(height(t),1); part(:)="other";
part(ismember(t.seed,trainSeeds(:)))="train";
part(ismember(t.seed,valSeeds(:)))="validation";
% Refuse accidental held-out leakage into fit tables.
t.partition=part;
end

function audit=auditCoverage(completion,actionRisk,experience,inventory,cfg,p4,p6,configs,seeds)
realized=completion;
if ismember('is_realized',completion.Properties.VariableNames)
    realized=completion(completion.is_realized==true,:);
end
feasCounts=struct();
if ~isempty(realized) && ismember('feasibility_class',realized.Properties.VariableNames)
    u=unique(string(realized.feasibility_class));
    for i=1:numel(u)
        feasCounts.(matlab.lang.makeValidName(char(u(i))))=sum(string(realized.feasibility_class)==u(i));
    end
end
nLocal=0; nOffOk=0; nOffFail=0; nHelpers=0;
if ~isempty(realized)
    nLocal=sum(string(realized.action)=="local");
    nOffOk=sum(string(realized.action)=="offload" & realized.completed_on_time==1);
    nOffFail=sum(string(realized.action)=="offload" & realized.completed_on_time==0);
    if ismember('processor_id',realized.Properties.VariableNames)
        off=realized(string(realized.action)=="offload",:);
        nHelpers=numel(unique(off.processor_id));
    end
end
nClassII=0; nClassIII=0;
if ~isempty(realized) && ismember('feasibility_class',realized.Properties.VariableNames)
    nClassII=sum(string(realized.feasibility_class)=="II");
    nClassIII=sum(string(realized.feasibility_class)=="III");
end
nCounterfactual=0;
if ismember('is_realized',completion.Properties.VariableNames)
    nCounterfactual=sum(completion.is_realized==false);
end
hasEpoch=false; hasAssign=false;
for i=1:numel(inventory)
    hasEpoch=hasEpoch || inventory(i).has_schedule_epochs;
    hasAssign=hasAssign || inventory(i).has_assignment_features;
end
% Per cfg/seed availability for P6 myopic and validation seeds
missingP6=strings(0,1); missingValP4=strings(0,1);
for ci=1:numel(configs)
    for si=1:numel(seeds)
        cfgId=configs(ci); seed=seeds(si);
        myopicComplete=isfile(fullfile(p6,sprintf('cfg%02d_seed%d',cfgId,seed),'CASE_COMPLETE.mat')) || ...
            isfile(fullfile(p6,sprintf('cfg%02d_seed%d',cfgId,seed),'risk_myopic_p6','COMPLETE.mat')) || ...
            isfolder(fullfile(p6,sprintf('cfg%02d_seed%d',cfgId,seed),'risk_myopic_p6','safety_offloading'));
        if ~myopicComplete
            missingP6(end+1)=sprintf('cfg%02d_seed%d',cfgId,seed); %#ok<AGROW>
        end
        if ismember(seed,cfg.sources.validation_seeds(:))
            p4case=fullfile(p4,sprintf('cfg%02d_seed%d',cfgId,seed),'deadline_local','safety_offloading','tasks.csv');
            if ~isfile(p4case)
                missingValP4(end+1)=sprintf('cfg%02d_seed%d',cfgId,seed); %#ok<AGROW>
            end
        end
    end
end
seqTransitions=0;
if ~isempty(experience) && ismember('next_decision_id',experience.Properties.VariableNames)
    seqTransitions=sum(isfinite(experience.next_decision_id));
end
audit=struct( ...
    'n_realized_rows',height(realized), ...
    'n_counterfactual_candidate_rows',nCounterfactual, ...
    'n_class_II',nClassII, ...
    'n_class_III',nClassIII, ...
    'n_local_actions',nLocal, ...
    'n_offload_success',nOffOk, ...
    'n_offload_fail',nOffFail, ...
    'n_distinct_helpers',nHelpers, ...
    'feasibility_counts',feasCounts, ...
    'has_assignment_features_any',hasAssign, ...
    'has_schedule_epochs_any',hasEpoch, ...
    'n_experience_rows',height(experience), ...
    'n_sequential_transitions',seqTransitions, ...
    'n_action_risk_rows',height(actionRisk), ...
    'missing_p6_instances',missingP6, ...
    'missing_validation_p4',missingValP4, ...
    'train_seeds',cfg.sources.train_seeds(:)', ...
    'validation_seeds',cfg.sources.validation_seeds(:)', ...
    'checks',struct( ...
        'data_partitioning',"train/validation by scenario seed; held-out reserved for P10", ...
        'action_coverage',"Classes II/III, helpers, offload success/fail, local", ...
        'safety_labels',"action_conditional_risk.csv vs policy_safety_costs.csv; no policy harm on actions"));
end

function d=collectDeficiencies(audit,cfg)
items={};
if audit.n_class_II<1
    items{end+1}=struct('code',"missing_class_II",'severity',"blocking", ...
        'detail',"No Class II realized actions in assembled completion table.");
end
if audit.n_class_III<1
    items{end+1}=struct('code',"missing_class_III",'severity',"blocking", ...
        'detail',"No Class III realized actions in assembled completion table.");
end
if audit.n_local_actions<1
    items{end+1}=struct('code',"missing_local_execution",'severity',"blocking", ...
        'detail',"No local-execution outcomes for completion/latency coverage.");
end
if audit.n_offload_success<1 || audit.n_offload_fail<1
    items{end+1}=struct('code',"incomplete_offload_outcomes",'severity',"blocking", ...
        'detail',"Need both successful and unsuccessful offloads.");
end
if audit.n_distinct_helpers<2
    items{end+1}=struct('code',"insufficient_helper_diversity",'severity',"warning", ...
        'detail',"Fewer than two distinct helpers observed among offloads.");
end
if ~audit.has_assignment_features_any
    items{end+1}=struct('code',"no_counterfactual_assignment_features",'severity',"warning", ...
        'detail',"P4 worlds lack assignment_features.csv; counterfactual action coverage depends on P6.");
end
if ~audit.has_schedule_epochs_any
    items{end+1}=struct('code',"no_schedule_epochs",'severity',"warning", ...
        'detail',"No schedule_epochs.csv yet; sequential DRL transitions use legacy decisions chains.");
end
if audit.n_sequential_transitions<1
    items{end+1}=struct('code',"no_sequential_transitions",'severity',"warning", ...
        'detail',"Experience table has no next_decision links.");
end
if audit.n_counterfactual_candidate_rows<1
    items{end+1}=struct('code',"no_counterfactual_candidates",'severity',"warning", ...
        'detail',"No unchosen assignment candidates; insufficient for counterfactual DRL coverage.");
end
if ~isempty(audit.missing_p6_instances)
    items{end+1}=struct('code',"p6_incomplete",'severity',"info", ...
        'detail',sprintf('P6 myopic worlds missing for %d cfg/seed instances (eval may still be running).', ...
        numel(audit.missing_p6_instances)));
end
nVal=numel(cfg.sources.validation_seeds);
if nVal>=1 && ~isempty(audit.missing_validation_p4)
    items{end+1}=struct('code',"validation_p4_missing",'severity',"blocking", ...
        'detail',"Validation seeds lack P4 deadline_local sources.");
end
% Flag validation seeds that still lack P6 myopic worlds.
valSeeds=cfg.sources.validation_seeds(:)';
p6ValMissing=0;
for i=1:numel(audit.missing_p6_instances)
    tok=regexp(char(audit.missing_p6_instances(i)),'seed(\d+)$','tokens','once');
    if ~isempty(tok) && ismember(str2double(tok{1}),valSeeds)
        p6ValMissing=p6ValMissing+1;
    end
end
if p6ValMissing>0
    items{end+1}=struct('code',"p6_validation_seed_gap",'severity',"warning", ...
        'detail',"Validation seeds lack risk_myopic_p6 worlds; calibrate P6-native features on train seeds only and defer val until seeds exist.");
end

blocking=0; warningN=0;
for i=1:numel(items)
    if strcmp(items{i}.severity,"blocking"), blocking=blocking+1; end
    if strcmp(items{i}.severity,"warning"), warningN=warningN+1; end
end
needsExtra=false;
for i=1:numel(items)
    if ismember(string(items{i}.code),["no_counterfactual_candidates","no_schedule_epochs", ...
            "no_sequential_transitions","no_counterfactual_assignment_features"])
        needsExtra=true;
    end
end
d=struct( ...
    'items',{items}, ...
    'blocking_count',blocking, ...
    'warning_count',warningN, ...
    'ready_for_calibration',blocking==0 && audit.n_realized_rows>0, ...
    'needs_additional_experience',needsExtra, ...
    'note',"Generate missing experience only after reviewing deficiencies; do not relaunch P4/P5.");
end
