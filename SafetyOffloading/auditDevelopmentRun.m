function [cpuAudit,failureAudit,summary] = auditDevelopmentRun(runRoot)
%AUDITDEVELOPMENTRUN Audit local CPU contention and offload latency stages.
root=fileparts(mfilename('fullpath')); addpath(root);
if nargin<1 || strlength(string(runRoot))==0
    candidates=dir(fullfile(root,'outputs','paired_unrestricted_rate12tps_density50_*'));
    candidates=candidates([candidates.isdir]);
    assert(~isempty(candidates),'Audit:MissingRun','No unrestricted development run found');
    [~,ix]=max([candidates.datenum]); runRoot=fullfile(candidates(ix).folder,candidates(ix).name);
end
runRoot=string(runRoot); cfg=loadConfig(fullfile(runRoot,'config.json'));
scenarioRoot=fullfile(runRoot,'scenario_0001');
baseTasks=readtable(fullfile(scenarioRoot,'no_offload','safety_offloading','tasks.csv'),'VariableNamingRule','preserve');
policyTasks=readtable(fullfile(scenarioRoot,'unrestricted','safety_offloading','tasks.csv'),'VariableNamingRule','preserve');
radio=readtable(fullfile(scenarioRoot,'unrestricted','safety_offloading','radio_events.csv'),'VariableNamingRule','preserve');
decisions=readtable(fullfile(scenarioRoot,'unrestricted','safety_offloading','decisions.csv'),'VariableNamingRule','preserve');
seed=baseTasks.scenario_id(1);

origins=unique(baseTasks.origin_id); cpuRows=cell(numel(origins),12);
for oi=1:numel(origins)
    origin=origins(oi); rows=baseTasks(baseTasks.origin_id==origin,:);
    rows=sortrows(rows,'release_ms');
    u=stableUniform(seed,"wilab-cpu",origin);
    cpuRate=cfg.compute.cpu_cycles_per_s_min+u*(cfg.compute.cpu_cycles_per_s_max-cfg.compute.cpu_cycles_per_s_min);
    serviceMs=cfg.compute.cycles_per_task/cpuRate*1000;
    arrivals=height(rows); spanS=max(1e-9,(max(rows.release_ms)-min(rows.release_ms))/1000);
    observedRate=max(0,arrivals-1)/spanS;
    queueDelay=rows.cpu_start_ms-rows.release_ms;
    overlap=false(height(rows),1);
    if height(rows)>1, overlap(2:end)=rows.release_ms(2:end)<rows.cpu_finish_ms(1:end-1); end
    expectedStart=nan(height(rows),1); lastFinish=-inf;
    for ri=1:height(rows)
        if isnan(rows.cpu_start_ms(ri)), continue; end
        expectedStart(ri)=max(rows.release_ms(ri),lastFinish);
        lastFinish=rows.cpu_finish_ms(ri);
    end
    persistenceError=max(abs(rows.cpu_start_ms-expectedStart),[],'omitnan');
    cpuRows(oi,:)={origin,cpuRate/1e9,serviceMs,arrivals,observedRate, ...
        observedRate*cfg.compute.cycles_per_task/cpuRate,sum(overlap), ...
        mean(queueDelay,'omitnan'),max(queueDelay,[],'omitnan'), ...
        sum(rows.completed_on_time),height(rows),persistenceError};
end
cpuAudit=cell2table(cpuRows,'VariableNames',{'origin_id','cpu_ghz','service_ms','arrivals', ...
    'observed_arrival_rate','observed_utilization','overlapping_arrivals','mean_queue_ms', ...
    'max_queue_ms','completed','tasks','cpu_persistence_error_ms'});

failureRows=cell(height(policyTasks),16);
for ti=1:height(policyTasks)
    t=policyTasks(ti,:); ev=radio(radio.task_id==t.task_id & string(radio.traffic_class)=="compute",:);
    inputEv=ev(string(ev.stage)=="input",:); resultEv=ev(string(ev.stage)=="result",:);
    inputFirst=min(inputEv.time_ms,[],'omitnan');
    resultFirst=min(resultEv.time_ms,[],'omitnan');
    if isempty(inputFirst), inputFirst=NaN; end
    if isempty(resultFirst), resultFirst=NaN; end
    inputScheduling=inputFirst-t.release_ms;
    inputTransfer=t.input_delivered_ms-inputFirst;
    helperQueue=t.cpu_start_ms-t.input_delivered_ms;
    computeTime=t.cpu_finish_ms-t.cpu_start_ms;
    returnScheduling=resultFirst-t.cpu_finish_ms;
    returnTransfer=t.result_delivered_ms-resultFirst;
    expectedFragments=ceil(t.in_bytes/cfg.radio.max_transport_block_bytes);
    targetDecoded=0; targetDecodedBeforeDeadline=0;
    if ~isempty(inputEv)
        targetDecoded=numel(unique(inputEv.fragment(inputEv.receiver_id==t.destination & inputEv.decoded==1)));
        targetDecodedBeforeDeadline=numel(unique(inputEv.fragment(inputEv.receiver_id==t.destination & ...
            inputEv.decoded==1 & inputEv.time_ms<=t.deadline_ms)));
    end
    if t.completed_on_time
        category="completed";
    elseif isnan(t.input_delivered_ms)
        if isempty(inputEv)
            category="input_scheduling";
        elseif inputFirst>t.deadline_ms
            category="input_scheduled_after_deadline";
        elseif targetDecodedBeforeDeadline<expectedFragments
            category="input_loss_or_late_fragment";
        else
            category="input_state_transition";
        end
    elseif isnan(t.cpu_start_ms)
        category="helper_queue";
    elseif isnan(t.cpu_finish_ms)
        category="helper_execution";
    elseif isnan(t.result_delivered_ms)
        if isempty(resultEv), category="result_scheduling"; else, category="result_loss"; end
    else
        category="late_end_to_end";
    end
    failureRows(ti,:)={t.task_id,t.origin_id,t.destination,t.release_ms,t.deadline_ms, ...
        inputScheduling,inputTransfer,helperQueue,computeTime,returnScheduling,returnTransfer, ...
        t.result_delivered_ms-t.release_ms,expectedFragments,targetDecoded,category,t.completed_on_time};
end
failureAudit=cell2table(failureRows,'VariableNames',{'task_id','origin_id','helper_id','release_ms', ...
    'deadline_ms','input_scheduling_ms','input_transfer_ms','helper_queue_ms','compute_ms', ...
    'return_scheduling_ms','return_transfer_ms','end_to_end_ms','expected_input_fragments', ...
    'decoded_input_fragments_at_helper','failure_category','completed_on_time'});
categories=groupcounts(failureAudit,'failure_category');
summary=table(height(baseTasks),sum(baseTasks.completed_on_time),height(policyTasks), ...
    sum(policyTasks.completed_on_time),min(cpuAudit.cpu_ghz),max(cpuAudit.cpu_ghz), ...
    sum(cpuAudit.overlapping_arrivals),max(cpuAudit.max_queue_ms), ...
    'VariableNames',{'local_tasks','local_completed','offload_tasks','offload_completed', ...
    'min_origin_cpu_ghz','max_origin_cpu_ghz','overlapping_arrivals','max_local_queue_ms'});
outDir=fullfile(runRoot,'implementation_audit'); if ~isfolder(outDir), mkdir(outDir); end
writetable(cpuAudit,fullfile(outDir,'cpu_audit.csv'));
writetable(failureAudit,fullfile(outDir,'offload_failure_decomposition.csv'));
writetable(categories,fullfile(outDir,'failure_categories.csv'));
writetable(summary,fullfile(outDir,'audit_summary.csv'));
save(fullfile(outDir,'audit.mat'),'cpuAudit','failureAudit','categories','summary','cfg','decisions');
fprintf('Development audit complete: %s\n',outDir); disp(summary); disp(categories);
end
