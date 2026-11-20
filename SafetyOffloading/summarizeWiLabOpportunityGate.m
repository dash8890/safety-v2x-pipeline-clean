function summary=summarizeWiLabOpportunityGate(runRoot)
%SUMMARIZEWILABOPPORTUNITYGATE Recompute global and causal-window metrics.
runRoot=string(runRoot);
methods=["local_only","instant_oracle","radio_oracle","safety_oracle","risk_aware"];
policies=["no_offload","oracle_instant","oracle_actual","safety_oracle","risk_heuristic"];
baseDir=fullfile(runRoot,methods(1),'safety_offloading');
baseSafety=readtable(fullfile(baseDir,'safety_packets.csv'),'VariableNamingRule','preserve');
baseTasks=readtable(fullfile(baseDir,'tasks.csv'),'VariableNamingRule','preserve');
baseCompleted=sum(baseTasks.completed_on_time); rows=cell(numel(methods),14);
for mi=1:numel(methods)
    caseDir=fullfile(runRoot,methods(mi),'safety_offloading');
    safety=readtable(fullfile(caseDir,'safety_packets.csv'),'VariableNamingRule','preserve');
    tasks=readtable(fullfile(caseDir,'tasks.csv'),'VariableNamingRule','preserve');
    completed=sum(tasks.completed_on_time); offloaded=sum(string(tasks.action)=="offload");
    if mi==1
        globalPrr=mean(safety.success_on_time); globalHarm=0; globalAvoided=0;
        windowExposures=0; windowHarmful=0; windowHarm=0;
    else
        paired=pairOutcomes(baseSafety,safety);
        globalPrr=mean(paired.policy_success); globalHarm=mean(paired.harmful_flip);
        globalAvoided=mean(paired.avoided_failure);
        decisions=readtable(fullfile(caseDir,'decisions.csv'),'VariableNamingRule','preserve');
        admitted=decisions(string(decisions.chosen_action)=="offload",:);
        inWindow=false(height(paired),1);
        for di=1:height(admitted)
            startMs=admitted.time_ms(di); deadlineMs=admitted.deadline_ms(di);
            inWindow=inWindow | (paired.created_ms>=startMs & paired.created_ms<startMs+deadlineMs);
        end
        windowExposures=sum(inWindow); windowHarmful=sum(paired.harmful_flip & inWindow);
        if windowExposures>0, windowHarm=windowHarmful/windowExposures; else, windowHarm=0; end
    end
    cfg=loadConfig(fullfile(runRoot,methods(mi),'config.json'));
    rows(mi,:)={methods(mi),policies(mi),completed,height(tasks),completed-baseCompleted,offloaded, ...
        globalPrr,globalHarm,globalAvoided,windowExposures,windowHarmful,windowHarm, ...
        windowHarm<=cfg.control.risk_harm_limit,fullfile(runRoot,methods(mi))};
end
summary=cell2table(rows,'VariableNames',{'method','policy','completed','tasks','completion_gain', ...
    'offloaded','global_safety_prr','global_harm_probability','global_avoided_failure_probability', ...
    'action_window_exposures','action_window_harmful_flips','action_window_harm_probability', ...
    'safety_constraint_met','output_dir'});
writetable(summary,fullfile(runRoot,'opportunity_summary.csv'));
save(fullfile(runRoot,'opportunity_summary.mat'),'summary');
disp(summary);
end
