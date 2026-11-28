function stationManagement = spv2xWiLabFinalize(stationManagement,outParams,simParams)
%SPV2XWILABFINALIZE Close outcomes and write the structured logging contract.
s=stationManagement.spv2x;
if startsWith(string(s.policy),"workflow_")
    stationManagement=spv2xWorkflowBridge('finalize',stationManagement,outParams,simParams);
    s=stationManagement.spv2x;
end
folder=fullfile(outParams.outputFolder,"safety_offloading");
if ~isfolder(folder), mkdir(folder); end
if startsWith(string(s.policy),"workflow_") && isempty(s.tasks)
    % Minimal legacy CSVs so paired tooling does not crash; real logs under workflow_v2/.
    writetable(table(),fullfile(folder,"tasks.csv"));
    writetable(table(),fullfile(folder,"decisions.csv"));
    if s.safety_row_cursor>0
        safety=array2table(s.safety_rows(1:s.safety_row_cursor,:),'VariableNames',{'flow_id','seq','tx_id','rx_id', ...
            'created_ms','deadline_ms','decoded_ms','success_on_time','bytes','range_m','distance_bin','packet_id'});
        safety.success_on_time=logical(safety.success_on_time);
        safety.scenario_id=repmat(simParams.seed,height(safety),1);
        safety.world=repmat(s.world,height(safety),1);
        safety=movevars(safety,{'scenario_id','world'},'Before',1);
        writetable(safety,fullfile(folder,"safety_packets.csv"));
    else
        writetable(table(),fullfile(folder,"safety_packets.csv"));
    end
    radio=cell2table(s.radio_rows,'VariableNames',{'scenario_id','time_ms','ue_id','packet_id','traffic_class', ...
        'task_id','stage','fragment','receiver_id','range_m','resource_id','decoded','world'});
    writetable(radio,fullfile(folder,"radio_events.csv"));
    stationManagement.spv2x=s;
    return
end
for i=1:numel(s.tasks)
    if ~ismember(s.tasks(i).status,["completed","rejected","late_cpu","late_result","expired_cpu_queue"])
        s.tasks(i).status="deadline_expired";
    end
end

safety=array2table(s.safety_rows(1:s.safety_row_cursor,:),'VariableNames',{'flow_id','seq','tx_id','rx_id', ...
    'created_ms','deadline_ms','decoded_ms','success_on_time','bytes','range_m','distance_bin','packet_id'});
safety.success_on_time=logical(safety.success_on_time); safety.scenario_id=repmat(simParams.seed,height(safety),1);
safety.world=repmat(s.world,height(safety),1); safety=movevars(safety,{'scenario_id','world'},'Before',1);

n=numel(s.tasks); taskRows=cell(n,19);
for i=1:n
    t=s.tasks(i); taskRows(i,:)={simParams.seed,t.task_id,t.origin,t.release_ms,t.deadline_ms,t.input_bytes, ...
        t.output_bytes,t.cycles,t.value,char(t.action),t.helper,t.decision_id,t.input_delivered_ms, ...
        t.cpu_start_ms,t.cpu_finish_ms,t.result_delivered_ms,char(t.status),t.status=="completed", ...
        t.input_fragments+t.output_fragments};
end
tasks=cell2table(taskRows,'VariableNames',{'scenario_id','task_id','origin_id','release_ms','deadline_ms', ...
    'in_bytes','out_bytes','cycles','value','action','destination','decision_id','input_delivered_ms', ...
    'cpu_start_ms','cpu_finish_ms','result_delivered_ms','failure_reason','completed_on_time','fragment_count'});
decisions=cell2table(s.decision_rows,'VariableNames',{'scenario_id','decision_id','time_ms','vehicle_id', ...
    'task_id','chosen_action','helper_id','helper_rank','risk_upper','cbr','candidate_helpers','helper_distance_m', ...
    'origin_queue_length','helper_queue_length','input_bytes','output_bytes','cycles','deadline_ms', ...
    'policy_id','world','predicted_local_finish_ms','predicted_best_remote_finish_ms', ...
    'predicted_local_slack_ms','predicted_best_remote_slack_ms','feasibility_class'});
radio=cell2table(s.radio_rows,'VariableNames',{'scenario_id','time_ms','ue_id','packet_id','traffic_class', ...
    'task_id','stage','fragment','receiver_id','range_m','resource_id','decoded','world'});
writetable(safety,fullfile(folder,"safety_packets.csv")); writetable(tasks,fullfile(folder,"tasks.csv"));
writetable(decisions,fullfile(folder,"decisions.csv")); writetable(radio,fullfile(folder,"radio_events.csv"));
active=find(s.cpu_nominal_rate>0);
role=repmat("helper",numel(active),1); role(ismember(active,s.origin_ids))="origin";
cpuProfiles=table(active,role,s.cpu_nominal_rate(active),s.cpu_background_utilization(active), ...
    s.cpu_rate(active),'VariableNames',{'vehicle_id','role','nominal_cycles_per_s', ...
    'background_utilization','effective_cycles_per_s'});
writetable(cpuProfiles,fullfile(folder,"cpu_profiles.csv"));

% P5 logging: completion rewards and schedule-epoch artifacts. Safety costs are
% logged separately from rewards and filled when a paired baseline exists.
if isfield(s,'reward_rows') && ~isempty(s.reward_rows)
    rewards=cell2table(s.reward_rows,'VariableNames', ...
        {'scenario_id','task_id','time_ms','reward','event'});
else
    % Completion-based reward can always be reconstructed from task outcomes.
    completed=find(arrayfun(@(t)t.status=="completed",s.tasks));
    rewardRows=cell(numel(completed),5);
    for i=1:numel(completed)
        t=s.tasks(completed(i));
        rewardRows(i,:)={simParams.seed,t.task_id,t.result_delivered_ms,t.value,"completion"};
    end
    rewards=cell2table(rewardRows,'VariableNames',{'scenario_id','task_id','time_ms','reward','event'});
end
writetable(rewards,fullfile(folder,"rewards.csv"));

if isfield(s,'safety_cost_rows') && ~isempty(s.safety_cost_rows)
    safetyCosts=cell2table(s.safety_cost_rows,'VariableNames', ...
        {'scenario_id','stratum','time_ms','harm_cost','failure_cost','source'});
else
    safetyCosts=cell2table(cell(0,6),'VariableNames', ...
        {'scenario_id','stratum','time_ms','harm_cost','failure_cost','source'});
end
writetable(safetyCosts,fullfile(folder,"safety_costs.csv"));

if isfield(s,'epoch_rows') && ~isempty(s.epoch_rows)
    epochs=cell2table(s.epoch_rows,'VariableNames', ...
        {'scenario_id','decision_id','epoch_ms','action_type','task_id','processor_id', ...
        'pending_tasks','candidate_assignments'});
    writetable(epochs,fullfile(folder,"schedule_epochs.csv"));
end
if isfield(s,'assignment_feature_rows') && ~isempty(s.assignment_feature_rows)
    assigns=cell2table(s.assignment_feature_rows,'VariableNames', ...
        {'scenario_id','decision_id','epoch_ms','task_id','processor_id','is_local', ...
        'processor_cpu_ghz','residual_workload_mcycles','estimated_input_latency_ms', ...
        'estimated_result_latency_ms','estimated_helper_queueing_ms', ...
        'estimated_computation_latency_ms','predicted_deadline_slack_ms', ...
        'predicted_completion_probability','predicted_conditional_safety_risk', ...
        'relative_distance_m','estimated_resource_block_slots','masked','mask_reason', ...
        'feasibility_class','chosen'});
    writetable(assigns,fullfile(folder,"assignment_features.csv"));
end
s.finalized=true; stationManagement.spv2x=s;
end

