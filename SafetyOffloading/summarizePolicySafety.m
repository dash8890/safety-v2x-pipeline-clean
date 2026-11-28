function [overall,byStratum,paired]=summarizePolicySafety(baseline,policy,harmLimit,failureLimit)
%SUMMARIZEPOLICYSAFETY Policy-horizon paired harm and absolute failure.
arguments
    baseline table
    policy table
    harmLimit (1,1) double = 0.01
    failureLimit (1,1) double = 0.10
end
paired=pairOutcomes(baseline,policy);
overall=metricRow(paired,"all",harmLimit,failureLimit);
bins=unique(paired.distance_bin); rows=cell(numel(bins),10);
for i=1:numel(bins)
    row=metricRow(paired(paired.distance_bin==bins(i),:),"distance_bin_"+string(bins(i)), ...
        harmLimit,failureLimit);
    rows(i,:)=table2cell(row);
end
byStratum=cell2table(rows,'VariableNames',overall.Properties.VariableNames);
end

function row=metricRow(paired,label,harmLimit,failureLimit)
n=height(paired); harmful=sum(paired.harmful_flip); avoided=sum(paired.avoided_failure);
policyFailures=sum(~paired.policy_success); baselineFailures=sum(~paired.baseline_success);
row=table(string(label),n,harmful,avoided,harmful/max(1,n),policyFailures/max(1,n), ...
    baselineFailures/max(1,n),mean(paired.policy_success), ...
    harmful/max(1,n)<=harmLimit,policyFailures/max(1,n)<=failureLimit, ...
    'VariableNames',{'stratum','opportunities','harmful_flips','avoided_failures', ...
    'harmful_failure_probability','absolute_failure_probability', ...
    'baseline_failure_probability','safety_prr','harm_constraint_met','failure_constraint_met'});
end
