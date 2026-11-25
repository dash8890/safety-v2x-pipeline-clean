function acts=WorkflowPolicyPrivileged(W)
%B5: Privileged diagnostic — uses true queues (same as queue-aware here) but
% labeled non-deployable. Small-instance: prefer remote B/C on fastest true CPUs.
acts=WorkflowPolicyQueueAware(W);
for i=1:numel(acts)
    acts(i).detail="privileged_"+string(acts(i).detail);
end
end
