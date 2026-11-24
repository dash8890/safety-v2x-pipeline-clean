function smoke_causal_policy()
root=fileparts(mfilename('fullpath'));
addpath(genpath(root));
W=WorkflowEngine('new',struct('origin_id',1,'n_vehicles',3,'cpu_hz',[2.8e8 4e9 0.8e9], ...
    'rel_deadline_s',0.35,'tb_bytes',1180,'ctrl_delay_s',0,'default_link_delay_s',0, ...
    'slot_s',5e-4,'max_admitted_cycles',2e8));
W.phased_values=true; W.value_high=12; W.value_low=1;
W.p_high_early=0.05; W.p_high_late=0.75; W.phase_split_s=0.16;
W.seed=300; W.horizon_s=1.45; W.policy="workflow_drop_low";
W.last_contact_change_t=zeros(1,3); W.last_adv_rx_t=-inf(1,3);
W.ddqn_weights_json=fullfile(root,'drl','outputs','ddqn_causal_dueling.weights.json');
[W,~]=WorkflowEngine('create_workflow',W,0.04);
assert(W.workflows(1).value==1 || W.workflows(1).value==12, 'bad value');
Wdrop=WorkflowEngine('drop_workflow',W,1);
assert(Wdrop.workflows(1).status=="EXPIRED", 'drop failed');
[obs,mask,~]=WorkflowCausalObs(W);
assert(numel(obs)==16 && nnz(mask)>=1, 'bad obs');
a=WorkflowPolicyCausalDDQN(W);
assert(~isempty(a), 'ddqn empty');
disp(a(1).type);
a2=WorkflowPolicyValueGreedy(W);
disp(a2(1).type);
a3=WorkflowPolicyDropLow(W);
disp(a3(1).type);
disp('SMOKE_OK');
end
