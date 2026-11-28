function [stationManagement,timeManagement] = spv2xWiLabInit(appParams,simParams,phyParams,outParams,simValues,stationManagement,timeManagement,positionManagement)
%SPV2XWILABINIT Initialize the application overlay on WiLabV2XSim state.
configPath = string(appParams.safetyOffloadingConfig);
if ~isfile(configPath)
    safetyRoot = fileparts(fileparts(mfilename('fullpath')));
    candidate = fullfile(fileparts(safetyRoot),configPath);
    if isfile(candidate), configPath = candidate;
    else, candidate = fullfile(safetyRoot,"config",extractAfter(configPath,"config/")); configPath = candidate;
    end
end
cfg = loadConfig(configPath);
ids = stationManagement.activeIDsCV2X(:);
maxID = simValues.maxID;

s.enabled = true;
s.cfg = cfg;
s.policy = string(appParams.safetyOffloadingPolicy);
s.world = string(appParams.safetyOffloadingWorld);
s.density = simParams.rho;
s.dqn = [];
s.risk_model = [];
s.log_radio_events = ~isfield(cfg,'logging') || ~isfield(cfg.logging,'radio_events') || cfg.logging.radio_events;
s.receiver_sample_fraction = 1;
if isfield(cfg.safety,'receiver_sample_fraction')
    s.receiver_sample_fraction = cfg.safety.receiver_sample_fraction;
end
if s.policy=="dqn"
    safetyRoot=fileparts(fileparts(mfilename('fullpath')));
    modelPath=string(cfg.control.dqn_model_path);
    if ~isfile(modelPath), modelPath=fullfile(safetyRoot,modelPath); end
    assert(isfile(modelPath),"DQN:MissingModel","DQN model not found: %s",modelPath);
    s.dqn=jsondecode(fileread(modelPath));
end
if s.policy=="risk_heuristic"
    safetyRoot=fileparts(fileparts(mfilename('fullpath')));
    modelPath=string(cfg.control.risk_model_path);
    if ~isfile(modelPath), modelPath=fullfile(safetyRoot,modelPath); end
    assert(isfile(modelPath),"SafetyOffloading:MissingRiskModel","Risk model not found: %s",modelPath);
    s.risk_model=jsondecode(fileread(modelPath));
end
% P7 calibrated estimators (shared by calibrated myopic and P8 learners; P6 unchanged).
s.p7_models=struct();
if ismember(s.policy,["risk_myopic_p7","masked_ddqn_p8","learn_assist_p8","traj_constrained_p8"])
    safetyRoot=fileparts(fileparts(mfilename('fullpath')));
    s.p7_models.action_risk=loadP7Json(safetyRoot,cfg,"action_risk_model_path","models/p7_action_risk_calibrated.json");
    s.p7_models.completion=loadP7Json(safetyRoot,cfg,"completion_model_path","models/p7_completion_calibrated.json");
    s.p7_models.latency=loadP7Json(safetyRoot,cfg,"latency_model_path","models/p7_latency_calibrated.json");
end
% P8 learning artifacts (optional; missing file => legal stub/fallback behavior).
s.p8_ddqn=struct(); s.p8_value=struct(); s.p8_traj=struct();
s.traj_exposure_spent=0; s.traj_time_frac_remaining=1;
if s.policy=="masked_ddqn_p8"
    safetyRoot=fileparts(fileparts(mfilename('fullpath')));
    s.p8_ddqn=loadP7Json(safetyRoot,cfg,"p8_ddqn_model_path","models/p8_masked_ddqn.json");
end
if s.policy=="learn_assist_p8"
    safetyRoot=fileparts(fileparts(mfilename('fullpath')));
    s.p8_value=loadP7Json(safetyRoot,cfg,"p8_value_model_path","models/p8_value_assist.json");
end
if s.policy=="traj_constrained_p8"
    safetyRoot=fileparts(fileparts(mfilename('fullpath')));
    s.p8_traj=loadP7Json(safetyRoot,cfg,"p8_traj_model_path","models/p8_trajectory_constraint.json");
end
s.next_tick_s = 0;
s.packet_counter = 0;
s.evaluation_range_m = min(cfg.safety.receiver_range_m,max(phyParams.Raw));
s.safety_seq = zeros(maxID,1);
s.pending_tx = cell(maxID,1);
s.meta_queue = cell(maxID,1);
s.cpu_queue = cell(maxID,1);
s.cpu_active = zeros(maxID,1);
s.cpu_finish_s = inf(maxID,1);

emptyMeta = struct('valid',false,'packet_id',0,'kind',"",'task_id',0, ...
    'stage',"",'fragment',0,'fragments',0,'src',0,'dst',0,'created_ms',0, ...
    'deadline_ms',0,'seq',0,'flow_id',0,'bytes',0);
s.current = repmat(emptyMeta,maxID,1);
s.empty_meta = emptyMeta;
estimatedPackets=ceil(simParams.simulationTime/(cfg.safety.generation_interval_ms/1000))*numel(ids);
estimatedReceivers=min(numel(ids)-1,max(1,ceil(3*simParams.rho*s.evaluation_range_m/1000)));
estimatedExposures=max(1024,ceil(estimatedPackets*estimatedReceivers*s.receiver_sample_fraction*1.10));
s.safety_rows=zeros(estimatedExposures,12);
s.safety_row_cursor=0;
s.radio_rows = cell(0,13);
s.decision_rows = cell(0,25);
s.output_folder = string(outParams.outputFolder);

if isfield(cfg.compute,'origin_ids')
    origins=double(cfg.compute.origin_ids(:));
    assert(all(ismember(origins,ids)),'SafetyOffloading:OriginOverride', ...
        'Every configured origin must be an active vehicle');
else
    originCount = max(1,round(numel(ids)*cfg.compute.origin_fraction));
    originIndex = unique(round(linspace(1,numel(ids),originCount)));
    origins = ids(originIndex);
end
s.origin_ids=origins(:);
s.helper_ids=setdiff(ids,origins);
s.cpu_nominal_rate=zeros(maxID,1);
s.cpu_background_utilization=zeros(maxID,1);
for v=ids'
    s.cpu_nominal_rate(v)=roleCpuRate(cfg.compute,simParams.seed,v,ismember(v,origins));
end
if isfield(cfg.compute,'helper_background_utilization')
    background=double(cfg.compute.helper_background_utilization);
    assert(isscalar(background) && background>=0 && background<1, ...
        'SafetyOffloading:BackgroundUtilization', ...
        'helper_background_utilization must be in [0,1)');
    s.cpu_background_utilization(s.helper_ids)=background;
end
if isfield(cfg.compute,'cpu_override_ids')
    overrideIDs=double(cfg.compute.cpu_override_ids(:));
    overrideRates=double(cfg.compute.cpu_override_cycles_per_s(:));
    assert(numel(overrideIDs)==numel(overrideRates), ...
        'SafetyOffloading:CPUOverride','CPU override IDs and rates must have equal length');
    for oi=1:numel(overrideIDs)
        assert(ismember(overrideIDs(oi),ids),'SafetyOffloading:CPUOverride', ...
            'CPU override vehicle %d is not active',overrideIDs(oi));
        s.cpu_nominal_rate(overrideIDs(oi))=overrideRates(oi);
    end
end
% Background work is modeled as a persistent reserved CPU share.  Every
% task sees the residual service rate, so helper contention is reproducible
% and does not reset at task decision epochs.
s.cpu_rate=s.cpu_nominal_rate.*(1-s.cpu_background_utilization);
releaseMs=[]; releaseOrigins=[];
if startsWith(s.policy,"workflow_")
    % Workflow arrivals are owned by WorkflowEngine (R3); skip binary task list.
    releaseMs=zeros(0,1); releaseOrigins=zeros(0,1);
elseif isfield(cfg.compute,'deterministic_release_ms')
    releaseMs=double(cfg.compute.deterministic_release_ms(:));
    if isfield(cfg.compute,'deterministic_origin_ids')
        releaseOrigins=double(cfg.compute.deterministic_origin_ids(:));
    else
        assert(isscalar(origins),'SafetyOffloading:DeterministicTasks', ...
            'deterministic_origin_ids is required with multiple origins');
        releaseOrigins=repmat(origins(1),numel(releaseMs),1);
    end
    assert(numel(releaseMs)==numel(releaseOrigins), ...
        'SafetyOffloading:DeterministicTasks','Release and origin arrays must have equal length');
    [releaseMs,order]=sort(releaseMs); releaseOrigins=releaseOrigins(order);
elseif isfield(cfg.compute,'arrival_model') && string(cfg.compute.arrival_model)=="poisson"
    assert(isfield(cfg.compute,'arrival_rate_tasks_per_s') && cfg.compute.arrival_rate_tasks_per_s>0, ...
        'SafetyOffloading:ArrivalRate','Poisson workload requires arrival_rate_tasks_per_s > 0');
    for o=1:numel(origins)
        taskStartMs=0; taskStopMs=simParams.simulationTime*1000;
        if isfield(cfg.compute,'task_start_ms'), taskStartMs=double(cfg.compute.task_start_ms); end
        if isfield(cfg.compute,'task_stop_ms'), taskStopMs=double(cfg.compute.task_stop_ms); end
        timeMs=taskStartMs; arrivalIndex=0;
        while true
            arrivalIndex=arrivalIndex+1;
            u=stableUniform(simParams.seed,"task-arrival",origins(o),arrivalIndex);
            timeMs=timeMs+(-log(max(u,realmin))/cfg.compute.arrival_rate_tasks_per_s)*1000;
            if timeMs>=min(taskStopMs,simParams.simulationTime*1000), break; end
            releaseMs(end+1,1)=timeMs; %#ok<AGROW>
            releaseOrigins(end+1,1)=origins(o); %#ok<AGROW>
        end
    end
    [releaseMs,order]=sort(releaseMs); releaseOrigins=releaseOrigins(order);
else
    epochs=(0:cfg.compute.arrival_interval_ms:(simParams.simulationTime*1000-1))';
    releaseMs=repelem(epochs,numel(origins));
    releaseOrigins=repmat(origins(:),numel(epochs),1);
end
nTasks = numel(releaseMs);
taskTemplate = struct('task_id',0,'origin',0,'release_ms',0,'deadline_ms',0, ...
    'input_bytes',0,'output_bytes',0,'cycles',0,'value',1,'action',"pending", ...
    'helper',0,'status',"pending",'input_fragments',0,'input_received',0, ...
    'output_fragments',0,'output_received',0,'cpu_owner',0,'cpu_start_ms',NaN, ...
    'cpu_finish_ms',NaN,'input_delivered_ms',NaN,'result_delivered_ms',NaN, ...
    'decision_id',0);
s.tasks = repmat(taskTemplate,nTasks,1);
k = 0;
for r = 1:nTasks
    k=k+1; t=taskTemplate; t.task_id=k; t.origin=releaseOrigins(r); t.release_ms=releaseMs(r);
    t.deadline_ms=releaseMs(r)+cfg.compute.deadline_ms; t.input_bytes=cfg.compute.input_bytes;
    t.output_bytes=cfg.compute.output_bytes; t.cycles=cfg.compute.cycles_per_task;
    s.tasks(k)=t;
end
s.safety_row_start=zeros(ceil(simParams.simulationTime/(cfg.safety.generation_interval_ms/1000))*numel(ids) + ...
    nTasks*(ceil(cfg.compute.input_bytes/cfg.radio.max_transport_block_bytes)+ ...
    ceil(cfg.compute.output_bytes/cfg.radio.max_transport_block_bytes)) + 16,1,'uint32');
s.safety_row_count=zeros(size(s.safety_row_start),'uint16');
s.release_cursor = 1;
s.decision_counter = 0;
s.scenario_id = simParams.seed;
s.position_snapshot = positionManagement;
s.pending_task_ids = [];
s.schedule_interval_ms = 5;
if isfield(cfg.control,'schedule_interval_ms')
    s.schedule_interval_ms = double(cfg.control.schedule_interval_ms);
end
s.next_schedule_ms = inf;
s.use_joint_scheduler = ismember(s.policy, ...
    ["joint_schedule","joint_myopic","schedule_myopic","joint_local","risk_myopic_p6","risk_myopic_p7","masked_ddqn_p8","learn_assist_p8","traj_constrained_p8"]);
s.epoch_rows = cell(0,8);
s.assignment_feature_rows = cell(0,21);
s.reward_rows = cell(0,5);
s.safety_cost_rows = cell(0,6);
s.offload_count = 0;
s.governor_spent = 0;
s.tentative_offloads = 0;
% R3 workflow overlay: attach hybrid engine (Mode-2 radio + WorkflowCpuQueue).
if startsWith(s.policy,"workflow_")
    addpath(genpath(fullfile(fileparts(fileparts(mfilename('fullpath'))),'workflow_v2')));
    s=spv2xWorkflowBridge('attach',s,simParams,positionManagement);
end
stationManagement.spv2x = s;
timeManagement.spv2xInitialized = true;
end

function model=loadP7Json(safetyRoot,cfg,fieldName,defaultRel)
model=struct();
rel=defaultRel;
if isfield(cfg,'control') && isfield(cfg.control,fieldName) && strlength(string(cfg.control.(fieldName)))>0
    rel=string(cfg.control.(fieldName));
end
path=rel;
if ~isfile(path), path=fullfile(safetyRoot,rel); end
if isfile(path), model=jsondecode(fileread(path)); end
end

function rate=roleCpuRate(compute,seed,vehicle,isOrigin)
if isOrigin && isfield(compute,'origin_cpu_ghz')
    range=double(compute.origin_cpu_ghz(:))*1e9;
elseif ~isOrigin && isfield(compute,'helper_cpu_ghz')
    range=double(compute.helper_cpu_ghz(:))*1e9;
else
    range=[double(compute.cpu_cycles_per_s_min);double(compute.cpu_cycles_per_s_max)];
end
assert(isscalar(range) || numel(range)==2,'SafetyOffloading:CpuRole', ...
    'CPU role configuration must contain one fixed value or a [min,max] range');
if isscalar(range)
    rate=range(1);
else
    assert(range(2)>=range(1),'SafetyOffloading:CpuRole','CPU role range must be ordered');
    u=stableUniform(seed,"wilab-cpu-role",vehicle,isOrigin);
    rate=range(1)+u*(range(2)-range(1));
end
end

