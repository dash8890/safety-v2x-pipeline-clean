function stationManagement = spv2xWiLabFinalize(stationManagement,outParams,simParams)
%SPV2XWILABFINALIZE Close outcomes and write the structured logging contract.
s=stationManagement.spv2x; folder=fullfile(outParams.outputFolder,"safety_offloading");
if ~isfolder(folder), mkdir(folder); end
for i=1:numel(s.tasks)
    if ~ismember(s.tasks(i).status,["completed","rejected","late_cpu","late_result","expired_cpu_queue"])
        s.tasks(i).status="deadline_expired";
    end
end

safety=array2table(s.safety_rows,'VariableNames',{'flow_id','seq','tx_id','rx_id', ...
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
    'policy_id','world'});
radio=cell2table(s.radio_rows,'VariableNames',{'scenario_id','time_ms','ue_id','packet_id','traffic_class', ...
    'task_id','stage','fragment','receiver_id','range_m','resource_id','decoded','world'});
writetable(safety,fullfile(folder,"safety_packets.csv")); writetable(tasks,fullfile(folder,"tasks.csv"));
writetable(decisions,fullfile(folder,"decisions.csv")); writetable(radio,fullfile(folder,"radio_events.csv"));
s.finalized=true; stationManagement.spv2x=s;
end

