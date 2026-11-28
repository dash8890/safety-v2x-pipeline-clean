function tentative=spv2xApplyAssignment(s,tentative,taskId,processorId,nowMs)
%SPV2XAPPLYASSIGNMENT Update tentative CPU/radio commitments after ASSIGN(i,j).
task=s.tasks(taskId); action="local";
if processorId~=task.origin, action="offload"; end
tentative.assigned_tasks(end+1)=taskId; %#ok<AGROW>
tentative.extra_cycles(processorId)=tentative.extra_cycles(processorId)+task.cycles;
if action=="offload"
    tentative.offload_count=tentative.offload_count+1;
end
entry=struct('task_id',taskId,'processor_id',processorId,'action',action,'time_ms',nowMs);
tentative.schedule(end+1,1)=entry; %#ok<AGROW>
end
