function obs=spv2xBuildObservation(s,pendingTaskIds,stationManagement,positionManagement,nowMs,tentative)
%SPV2XBUILDOBSERVATION Structured P5 observation groups A-D for one epoch.
cfg=s.cfg; ids=unique([s.origin_ids(:); s.helper_ids(:)]);
obs=struct();
obs.time_ms=nowMs;
obs.pending_task_ids=pendingTaskIds(:)';
obs.tasks=repmat(emptyTask(),0,1);
obs.assignments=repmat(emptyAssignment(),0,1);
obs.origins=repmat(emptyOrigin(),0,1);
obs.comm=commSafetyFeatures(s,stationManagement,nowMs);

for ti=1:numel(pendingTaskIds)
    tid=pendingTaskIds(ti); task=s.tasks(tid); origin=task.origin;
    helpers=helperCandidates(origin,ids,positionManagement,cfg.compute.helper_range_m,cfg.compute.max_helpers);
    processors=[origin; helpers(:)];
    [feasibility,localSlack,remoteSlack,localFinish,remoteFinish]=spv2xFeasibilityClass( ...
        s,tid,helpers,nowMs,tentative);
    ageMs=nowMs-task.release_ms;
    localQueueMs=max(0,localFinish-(nowMs+task.cycles/s.cpu_rate(origin)*1000));
    tFeat=emptyTask();
    tFeat.task_id=tid; tFeat.origin=origin; tFeat.remaining_deadline_ms=task.deadline_ms-nowMs;
    tFeat.age_ms=ageMs; tFeat.cycles_mcycles=task.cycles/1e6;
    tFeat.input_bytes=task.input_bytes; tFeat.output_bytes=task.output_bytes;
    tFeat.local_processing_ms=task.cycles/s.cpu_rate(origin)*1000;
    tFeat.local_queueing_ms=localQueueMs; tFeat.local_deadline_slack_ms=localSlack;
    tFeat.required_input_blocks=ceil(task.input_bytes/cfg.radio.max_transport_block_bytes);
    tFeat.required_result_blocks=ceil(task.output_bytes/cfg.radio.max_transport_block_bytes);
    tFeat.local_feasible=localSlack>=0; tFeat.feasibility_class=feasibility;
    tFeat.best_remote_slack_ms=remoteSlack; tFeat.best_remote_finish_ms=remoteFinish;
    obs.tasks(end+1,1)=tFeat; %#ok<AGROW>

    for pi=1:numel(processors)
        j=processors(pi); isLocal=(j==origin);
        aFeat=emptyAssignment();
        aFeat.task_id=tid; aFeat.processor_id=j; aFeat.is_local=isLocal;
        aFeat.processor_cpu_ghz=s.cpu_rate(j)/1e9;
        aFeat.residual_workload_mcycles=residualWorkloadMcycles(s,j,nowMs/1000,tentative);
        aFeat.workload_age_ms=0; % instantaneous local state in the simulator
        if isLocal
            estIn=0; estOut=0; estQueue=localQueueMs; estComp=tFeat.local_processing_ms;
            predFinish=localFinish; distance=0; relVel=0;
        else
            fragmentsIn=ceil(task.input_bytes/cfg.radio.max_transport_block_bytes);
            fragmentsOut=ceil(task.output_bytes/cfg.radio.max_transport_block_bytes);
            estIn=fragmentsIn*cfg.radio.selection_window_ms;
            estOut=fragmentsOut*cfg.radio.selection_window_ms;
            helperFinish=spv2xForecastCpuFinishMs(s,j,nowMs/1000,task.cycles,tentative);
            estComp=task.cycles/s.cpu_rate(j)*1000;
            estQueue=max(0,helperFinish-(nowMs+estComp));
            predFinish=helperFinish+estIn+estOut;
            distance=positionManagement.distanceReal(origin,j);
            relVel=0;
            if isfield(positionManagement,'XvehicleReal') && isfield(positionManagement,'YvehicleReal')
                % Placeholder: mobility vectors are not always exposed; distance remains primary.
                relVel=NaN;
            end
        end
        slack=task.deadline_ms-predFinish;
        aFeat.estimated_input_latency_ms=estIn;
        aFeat.estimated_result_latency_ms=estOut;
        aFeat.estimated_helper_queueing_ms=estQueue;
        aFeat.estimated_computation_latency_ms=estComp;
        aFeat.predicted_remote_deadline_slack_ms=slack;
        aFeat.predicted_completion_probability=double(slack>=0);
        aFeat.predicted_conditional_safety_risk=conditionalRisk(s,nowMs,isLocal);
        aFeat.relative_distance_m=distance;
        aFeat.relative_velocity_mps=relVel;
        aFeat.estimated_resource_block_slots= ...
            (ceil(task.input_bytes/cfg.radio.max_transport_block_bytes)+ ...
             ceil(task.output_bytes/cfg.radio.max_transport_block_bytes))* ...
            max(1,round(cfg.radio.selection_window_ms/cfg.radio.slot_duration_ms));
        aFeat.masked=false;
        aFeat.mask_reason="";
        if isLocal
            aFeat.workload_source="local_exact";
        else
            aFeat.workload_source="helper_privileged";
        end
        aFeat=spv2xApplyP7Calibration(s,aFeat);
        obs.assignments(end+1,1)=aFeat; %#ok<AGROW>
    end
end

originIds=unique(arrayfun(@(t)t.origin,obs.tasks));
for oi=1:numel(originIds)
    o=originIds(oi);
    pendingHere=sum([obs.tasks.origin]==o);
    oFeat=emptyOrigin();
    oFeat.origin_id=o; oFeat.pending_task_count=pendingHere;
    oFeat.arrival_rate_estimate=cfg.compute.arrival_rate_tasks_per_s;
    if isfield(cfg.compute,'arrival_rate_tasks_per_s')
        oFeat.arrival_rate_estimate=cfg.compute.arrival_rate_tasks_per_s;
    else
        oFeat.arrival_rate_estimate=NaN;
    end
    util=0;
    if s.cpu_active(o)>0, util=1; end
    oFeat.cpu_utilization=util;
    oFeat.remaining_committed_workload_mcycles=residualWorkloadMcycles(s,o,nowMs/1000,tentative);
    helpers=helperCandidates(o,ids,positionManagement,cfg.compute.helper_range_m,cfg.compute.max_helpers);
    oFeat.eligible_helper_count=numel(helpers);
    oFeat.aggregate_helper_capacity_ghz=sum(s.cpu_rate(helpers))/1e9;
    obs.origins(end+1,1)=oFeat; %#ok<AGROW>
end
end

function w=residualWorkloadMcycles(s,vehicle,nowS,tentative)
finish=spv2xForecastCpuFinishMs(s,vehicle,nowS,0,tentative);
w=max(0,(finish-nowS*1000)/1000*s.cpu_rate(vehicle)/1e6);
end

function risk=conditionalRisk(s,nowMs,isLocal)
if isLocal, risk=0; return; end
cfg=s.cfg; base=min(1,0.002+0.003*(cfg.compute.input_bytes/cfg.radio.max_transport_block_bytes));
committed=0; windowMs=cfg.compute.deadline_ms;
for k=1:numel(s.tasks)
    if s.tasks(k).action~="offload", continue; end
    if abs(s.tasks(k).release_ms-nowMs)<=windowMs, committed=committed+1; end
end
if isfield(s,'tentative_offloads'), committed=committed+s.tentative_offloads; end
risk=min(1,base*(1+committed));
end

function feat=commSafetyFeatures(s,stationManagement,nowMs)
feat=struct('recent_cbr',0,'cbr_trend',0,'sensed_resource_availability',NaN, ...
    'recent_scheduling_delay_ms',NaN,'recent_delivery_success',NaN, ...
    'outstanding_input_blocks',0,'outstanding_result_blocks',0, ...
    'estimated_safety_externality',0,'time_ms',nowMs);
if isfield(stationManagement,'cbrCV2Xvalues') && ~isempty(s.origin_ids)
    values=stationManagement.cbrCV2Xvalues(s.origin_ids(1),:); values=values(values>=0);
    if ~isempty(values)
        feat.recent_cbr=values(end);
        if numel(values)>=2, feat.cbr_trend=values(end)-values(end-1); end
    end
end
for v=1:numel(s.pending_tx)
    for k=1:numel(s.pending_tx{v})
        item=s.pending_tx{v}(k);
        if item.stage=="input", feat.outstanding_input_blocks=feat.outstanding_input_blocks+1; end
        if item.stage=="result", feat.outstanding_result_blocks=feat.outstanding_result_blocks+1; end
    end
end
feat.estimated_safety_externality=feat.recent_cbr*0.01+0.001*feat.outstanding_input_blocks;
end

function candidates=helperCandidates(origin,ids,positionManagement,maxRange,maxHelpers)
if ~isfield(positionManagement,'distanceReal'), candidates=[]; return; end
d=positionManagement.distanceReal(origin,ids); valid=ids(d>0 & d<=maxRange);
if isempty(valid), candidates=[]; return; end
[~,order]=sort(positionManagement.distanceReal(origin,valid)); valid=valid(order);
candidates=valid(1:min(maxHelpers,numel(valid)));
end

function t=emptyTask()
t=struct('task_id',0,'origin',0,'remaining_deadline_ms',0,'age_ms',0,'cycles_mcycles',0, ...
    'input_bytes',0,'output_bytes',0,'local_processing_ms',0,'local_queueing_ms',0, ...
    'local_deadline_slack_ms',0,'required_input_blocks',0,'required_result_blocks',0, ...
    'local_feasible',false,'feasibility_class',"IV",'best_remote_slack_ms',-inf, ...
    'best_remote_finish_ms',inf);
end

function a=emptyAssignment()
a=struct('task_id',0,'processor_id',0,'is_local',false,'processor_cpu_ghz',0, ...
    'residual_workload_mcycles',0,'workload_age_ms',0,'estimated_input_latency_ms',0, ...
    'estimated_result_latency_ms',0,'estimated_helper_queueing_ms',0, ...
    'estimated_computation_latency_ms',0,'predicted_remote_deadline_slack_ms',0, ...
    'predicted_completion_probability',0,'predicted_conditional_safety_risk',0, ...
    'relative_distance_m',0,'relative_velocity_mps',0,'estimated_resource_block_slots',0, ...
    'masked',false,'mask_reason',"",'workload_source',"unknown");
end

function o=emptyOrigin()
o=struct('origin_id',0,'pending_task_count',0,'arrival_rate_estimate',NaN, ...
    'cpu_utilization',0,'remaining_committed_workload_mcycles',0, ...
    'eligible_helper_count',0,'aggregate_helper_capacity_ghz',0, ...
    'capacity_source',"simulator");
end
