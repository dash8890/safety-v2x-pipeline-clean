function [costs,overall,byStratum]=spv2xRecordSafetyCosts(baselineSafety,policySafety,harmLimit,failureLimit,scenarioId,timeMs)
%SPV2XRECORDSAFETYCOSTS Policy-level safety costs consistent with P2 accounting.
% c_H and c_F match summarizePolicySafety / roadmap §8.7 pooled constraints:
%   E[sum (H_e - delta)] <= 0,  E[sum (1-Y_e - epsilon)] <= 0
% reported here as realized totals over the matched exposure set.
if nargin<3 || isempty(harmLimit), harmLimit=0.01; end
if nargin<4 || isempty(failureLimit), failureLimit=0.10; end
if nargin<5 || isempty(scenarioId), scenarioId=0; end
if nargin<6 || isempty(timeMs), timeMs=NaN; end
[overall,byStratum]=summarizePolicySafety(baselineSafety,policySafety,harmLimit,failureLimit);
n=overall.opportunities;
harmCost=overall.harmful_flips - harmLimit*n;
failCost=n*overall.absolute_failure_probability - failureLimit*n;
costs=table(scenarioId,timeMs,string("all"),harmCost,failCost,n, ...
    overall.harmful_failure_probability,overall.absolute_failure_probability, ...
    'VariableNames',{'scenario_id','time_ms','stratum','harm_cost','failure_cost', ...
    'opportunities','harmful_failure_probability','absolute_failure_probability'});
for i=1:height(byStratum)
    ni=byStratum.opportunities(i);
    costs=[costs; {scenarioId,timeMs,byStratum.stratum(i), ...
        byStratum.harmful_flips(i)-harmLimit*ni, ...
        ni*byStratum.absolute_failure_probability(i)-failureLimit*ni, ...
        ni,byStratum.harmful_failure_probability(i),byStratum.absolute_failure_probability(i)}]; %#ok<AGROW>
end
end
