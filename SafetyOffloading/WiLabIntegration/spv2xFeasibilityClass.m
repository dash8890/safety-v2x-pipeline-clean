function [feasibility,localSlack,remoteSlack,localFinishMs,remoteFinishMs]=spv2xFeasibilityClass( ...
    s,taskId,helperIds,nowMs,tentative)
%SPV2XFEASIBILITYCLASS Predictive Class I-IV from local/remote deadline slack.
task=s.tasks(taskId); nowS=nowMs/1000;
localFinishMs=spv2xForecastCpuFinishMs(s,task.origin,nowS,task.cycles,tentative);
localSlack=task.deadline_ms-localFinishMs;
remoteFinishMs=-inf; remoteSlack=-inf;
if ~isempty(helperIds)
    finishes=zeros(numel(helperIds),1);
    for i=1:numel(helperIds)
        helperFinish=spv2xForecastCpuFinishMs(s,helperIds(i),nowS,task.cycles,tentative);
        fragments=ceil(task.input_bytes/s.cfg.radio.max_transport_block_bytes)+ ...
            ceil(task.output_bytes/s.cfg.radio.max_transport_block_bytes);
        finishes(i)=helperFinish+fragments*s.cfg.radio.selection_window_ms;
    end
    remoteFinishMs=min(finishes);
    remoteSlack=task.deadline_ms-remoteFinishMs;
end
if localSlack>=0 && remoteSlack<0
    feasibility="I";
elseif localSlack>=0 && remoteSlack>=0
    feasibility="II";
elseif localSlack<0 && remoteSlack>=0
    feasibility="III";
else
    feasibility="IV";
end
end
