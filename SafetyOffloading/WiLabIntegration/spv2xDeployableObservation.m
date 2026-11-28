function dep=spv2xDeployableObservation(obs)
%SPV2XDEPLOYABLEOBSERVATION Strip simulator-privileged fields from O_t.
% Helper residual workloads are not free local observations.  Until an
% explicit announcement channel exists, deployable remote residuals are NaN
% and remote completion uses communication+compute estimates with unknown
% helper queue (treated as zero only as an explicit estimation assumption,
% recorded in workload_source).
dep=obs;
if ~isfield(dep,'assignments') || isempty(dep.assignments), return; end
for i=1:numel(dep.assignments)
    a=dep.assignments(i);
    if a.is_local
        a.workload_source="local_exact";
    else
        a.residual_workload_mcycles=NaN;
        a.workload_age_ms=NaN;
        a.estimated_helper_queueing_ms=NaN;
        a.workload_source="helper_unannounced";
        % Recompute slack without privileged helper backlog: compute + link only.
        comp=a.estimated_computation_latency_ms;
        link=a.estimated_input_latency_ms+a.estimated_result_latency_ms;
        % predicted_remote_deadline_slack_ms was absolute; recover deadline via
        % finish = now is unavailable here, so adjust slack by removing queued
        % component that was embedded in the privileged estimate when present.
        if isfield(obs.assignments,'estimated_helper_queueing_ms')
            qPriv=obs.assignments(i).estimated_helper_queueing_ms;
            if isfinite(qPriv)
                a.predicted_remote_deadline_slack_ms=a.predicted_remote_deadline_slack_ms+qPriv;
            end
        end
        a.predicted_completion_probability=double(a.predicted_remote_deadline_slack_ms>=0);
        a.estimated_computation_latency_ms=comp;
        a.estimated_input_latency_ms=a.estimated_input_latency_ms;
        a.estimated_result_latency_ms=a.estimated_result_latency_ms;
        a.estimated_helper_queueing_ms=0; % estimation assumption, not observed truth
    end
    dep.assignments(i)=a;
end
% Origin features for foreign vehicles are not locally known.
if isfield(dep,'origins')
    for i=1:numel(dep.origins)
        dep.origins(i).aggregate_helper_capacity_ghz=NaN;
        dep.origins(i).capacity_source="unannounced";
    end
end
end
