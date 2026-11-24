function acts=WorkflowPolicyJointImmediate(W)
%B4 placement with immediate release (R3-M release ablation).
% Same pickJoint as B4; release READY_HELD remote edges as soon as dest committed
% / contact allows without additional HOLD delay policy.
W.release_mode="immediate";
acts=WorkflowPolicyJointPlanner(W);
for i=1:numel(acts)
    if acts(i).type=="RELEASE_EDGE"
        acts(i).detail="immediate_"+string(acts(i).detail);
    end
end
end
