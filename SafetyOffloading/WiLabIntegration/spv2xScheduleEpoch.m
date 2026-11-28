function [s,epochLog]=spv2xScheduleEpoch(s,pendingTaskIds,stationManagement,positionManagement,nowMs,policy)
%SPV2XSCHEDULEEPOCH Sequential joint schedule construction (ASSIGN/REJECT/STOP).
% Does not advance WiLab time. Commits are applied to s after STOP.
pendingTaskIds=pendingTaskIds(:)';
tentative=spv2xNewTentativeState(s);
epochLog=struct('epoch_ms',nowMs,'actions',{{}},'observations',{{}},'observations_full',{{}});
myopicPlan=struct('type',"STOP",'task_id',0,'processor_id',0); myopicPlan=myopicPlan(1:0);
maxSteps=numel(pendingTaskIds)+1;
for step=1:maxSteps
    remaining=setdiff(pendingTaskIds,[tentative.assigned_tasks(:); tentative.rejected_tasks(:)],'stable');
    obsFull=spv2xBuildObservation(s,remaining,stationManagement,positionManagement,nowMs,tentative);
    s.tentative_offloads=tentative.offload_count;
    obsFull=spv2xMaskActions(obsFull,s,tentative);
    % Action selection uses the deployable observation (no helper-queue leakage).
    obs=spv2xDeployableObservation(obsFull);
    obs=spv2xMaskActions(obs,s,tentative);
    [action,myopicPlan]=selectAction(obs,policy,s,tentative,myopicPlan);
    epochLog.actions{end+1}=action; %#ok<AGROW>
    epochLog.observations{end+1}=obs; %#ok<AGROW>
    epochLog.observations_full{end+1}=obsFull; %#ok<AGROW>
    if ~isstruct(action) || ~isfield(action,'type') || isempty(action.type)
        break
    end
    atype=lower(string(action.type)); atype=atype(1);
    if atype=="stop" || atype=="defer" || isempty(remaining)
        % DEFER leaves the task pending_unassigned; Tick arms a finite wake.
        break
    elseif atype=="reject"
        tentative.rejected_tasks(end+1)=action.task_id; %#ok<AGROW>
    elseif atype=="assign"
        if isempty(obs.assignments)
            break
        end
        hit=([obs.assignments.task_id]==action.task_id) & ...
            ([obs.assignments.processor_id]==action.processor_id) & ...
            [obs.assignments.masked];
        if any(hit)
            break
        end
        tentative=spv2xApplyAssignment(s,tentative,action.task_id,action.processor_id,nowMs);
    else
        break
    end
end
s=commitSchedule(s,tentative,nowMs);
end

function [action,myopicPlan]=selectAction(obs,policy,s,tentative,myopicPlan)
action=struct('type',"STOP",'task_id',0,'processor_id',0);
if isempty(obs.tasks), return; end
policyId=string(policy);
useMyopic=ismember(policyId,["risk_myopic_p6","risk_myopic_p7","joint_myopic","joint_schedule","schedule_myopic"]);
if policyId=="joint_local"
    for i=1:numel(obs.assignments)
        a=obs.assignments(i);
        if a.is_local && ~a.masked
            action=struct('type',"ASSIGN",'task_id',a.task_id,'processor_id',a.processor_id);
            return
        end
    end
    action=struct('type',"REJECT",'task_id',obs.tasks(1).task_id,'processor_id',0);
elseif policyId=="masked_ddqn_p8"
    % Alternative A: one ASSIGN/REJECT/STOP decision per sequential step.
    action=spv2xSelectMaskedDdqn(obs,s,tentative);
elseif policyId=="learn_assist_p8"
    % Alternative B: frozen optimizer + learned future-value term.
    if isempty(myopicPlan)
        myopicPlan=spv2xLearningAssistedMyopicPlan(obs,s,tentative);
    end
    if isempty(myopicPlan)
        action=struct('type',"STOP",'task_id',0,'processor_id',0);
    else
        action=myopicPlan(1);
        myopicPlan=myopicPlan(2:end);
        if ~isfield(action,'type') || ~validActionType(action.type)
            action=struct('type',"STOP",'task_id',0,'processor_id',0);
            myopicPlan=myopicPlan(1:0);
        end
    end
elseif policyId=="traj_constrained_p8"
    % P8.C: trajectory-aware cumulative exposure gate (LOCAL/OFFLOAD/DEFER/REJECT).
    if isempty(myopicPlan)
        myopicPlan=spv2xTrajectoryConstrainedPlan(obs,s,tentative);
    end
    if isempty(myopicPlan)
        action=struct('type',"STOP",'task_id',0,'processor_id',0);
    else
        action=myopicPlan(1);
        myopicPlan=myopicPlan(2:end);
        if ~isfield(action,'type') || ~validActionType(action.type)
            action=struct('type',"STOP",'task_id',0,'processor_id',0);
            myopicPlan=myopicPlan(1:0);
        end
    end
elseif useMyopic
    if isempty(myopicPlan)
        myopicPlan=spv2xRiskConstrainedMyopicPlan(obs,s,tentative);
    end
    if isempty(myopicPlan)
        action=struct('type',"STOP",'task_id',0,'processor_id',0);
    else
        action=myopicPlan(1);
        myopicPlan=myopicPlan(2:end);
    end
else
    % Unknown joint policy: stop rather than silently rewriting to myopic.
    action=struct('type',"STOP",'task_id',0,'processor_id',0);
end
end

function tf=validActionType(t)
st=string(t);
tf=isscalar(st) && strlength(strtrim(st))>0;
end

function s=commitSchedule(s,tentative,nowMs)
for i=1:numel(tentative.schedule)
    e=tentative.schedule(i); tid=e.task_id; proc=e.processor_id;
    s.tasks(tid).action=e.action; s.tasks(tid).helper=0;
    if e.action=="offload", s.tasks(tid).helper=proc; s.offload_count=s.offload_count+1; end
    s.tasks(tid).cpu_owner=proc;
    if e.action=="local"
        s.tasks(tid).status="queued_local"; s.cpu_queue{proc}(end+1)=tid;
    else
        ideal=isfield(s.cfg.compute,'ideal_communication') && s.cfg.compute.ideal_communication;
        if ideal
            s.tasks(tid).input_delivered_ms=nowMs; s.tasks(tid).status="queued_remote";
            s.cpu_queue{proc}(end+1)=tid;
        else
            origin=s.tasks(tid).origin;
            [s,items]=makeFragmentsLocal(s,tid,origin,proc,s.tasks(tid).input_bytes,"input",nowMs);
            s.tasks(tid).input_fragments=numel(items);
            s.tasks(tid).status="input_in_flight";
            s.pending_tx{origin}=[s.pending_tx{origin};items];
        end
    end
end
for tid=tentative.rejected_tasks(:)'
    s.tasks(tid).action="reject"; s.tasks(tid).status="rejected";
end
end

function [s,items]=makeFragmentsLocal(s,tid,src,dst,bytes,stage,createdMs)
n=ceil(bytes/s.cfg.radio.max_transport_block_bytes); items=repmat(s.empty_meta,n,1);
for f=1:n
    s.packet_counter=s.packet_counter+1; m=s.empty_meta; m.valid=true; m.packet_id=s.packet_counter;
    m.kind="compute"; m.task_id=tid; m.stage=stage; m.fragment=f; m.fragments=n;
    m.src=src; m.dst=dst; m.created_ms=createdMs; m.deadline_ms=s.tasks(tid).deadline_ms;
    m.seq=f; m.flow_id=1000000+tid;
    m.bytes=min(s.cfg.radio.max_transport_block_bytes,bytes-(f-1)*s.cfg.radio.max_transport_block_bytes);
    items(f)=m;
end
end
