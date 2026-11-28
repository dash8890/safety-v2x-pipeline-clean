function obs=spv2xMaskActions(obs,s,tentative)
%SPV2XMASKACTIONS Physically impossible or dominated assignments are masked.
% Mask rules (P5):
%   - processor capacity missing / zero rate
%   - remote assignment with no helper id
%   - predicted finish already past deadline by a hard margin and local also infeasible
%   - duplicate assignment of an already committed task
for i=1:numel(obs.assignments)
    a=obs.assignments(i); reason="";
    if a.processor_id<=0 || s.cpu_rate(a.processor_id)<=0
        reason="invalid_processor";
    elseif ~a.is_local && a.processor_id==s.tasks(a.task_id).origin
        reason="remote_requires_helper";
    elseif isfield(tentative,'assigned_tasks') && ismember(a.task_id,tentative.assigned_tasks)
        reason="task_already_assigned";
    elseif a.predicted_remote_deadline_slack_ms<-s.cfg.compute.deadline_ms
        reason="hard_deadline_violation";
    end
    if reason~=""
        a.masked=true; a.mask_reason=reason;
    else
        a.masked=false; a.mask_reason="";
    end
    obs.assignments(i)=a;
end
end
