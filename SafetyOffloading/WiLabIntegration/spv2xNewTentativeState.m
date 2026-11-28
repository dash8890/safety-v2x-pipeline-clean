function tentative=spv2xNewTentativeState(s)
%SPV2XNEWTENTATIVESTATE Snapshot overlays for sequential schedule construction.
tentative=struct();
tentative.cpu_queue=s.cpu_queue;
tentative.cpu_active=s.cpu_active;
tentative.cpu_finish_s=s.cpu_finish_s;
tentative.extra_cycles=zeros(size(s.cpu_rate));
tentative.assigned_tasks=[];
tentative.offload_count=0;
tentative.rejected_tasks=[];
tentative.schedule=repmat(struct('task_id',0,'processor_id',0,'action',"",'time_ms',0),0,1);
end
