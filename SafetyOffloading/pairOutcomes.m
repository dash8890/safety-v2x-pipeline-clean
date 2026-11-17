function paired = pairOutcomes(baseline, policy)
%PAIROUTCOMES Join safety opportunities and calculate causal outcome labels.
keys = {'flow_id','seq','tx_id','rx_id'};
B = baseline(:, [keys, {'created_ms','success_on_time','distance_bin'}]);
P = policy(:, [keys, {'success_on_time'}]);
B.Properties.VariableNames{strcmp(B.Properties.VariableNames,'success_on_time')} = 'baseline_success';
P.Properties.VariableNames{end} = 'policy_success';
paired = innerjoin(B, P, 'Keys', keys);
assert(height(paired) == height(B) && height(paired) == height(P), ...
    "Pairing:ExposureMismatch", "Both worlds must contain exactly the same safety exposures");
paired.harmful_flip = paired.baseline_success & ~paired.policy_success;
paired.avoided_failure = ~paired.baseline_success & paired.policy_success;
end

