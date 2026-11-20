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
s.next_tick_s = 0;
s.packet_counter = 0;
s.evaluation_range_m = min(cfg.safety.receiver_range_m,max(phyParams.Raw));
s.safety_seq = zeros(maxID,1);
s.pending_tx = cell(maxID,1);
s.meta_queue = cell(maxID,1);
s.cpu_queue = cell(maxID,1);
s.cpu_active = zeros(maxID,1);
s.cpu_finish_s = inf(maxID,1);
s.cpu_rate = zeros(maxID,1);
for v = ids'
    u = stableUniform(simParams.seed,"wilab-cpu",v);
    s.cpu_rate(v) = cfg.compute.cpu_cycles_per_s_min + ...
        u*(cfg.compute.cpu_cycles_per_s_max-cfg.compute.cpu_cycles_per_s_min);
end
if isfield(cfg.compute,'cpu_override_ids')
    overrideIDs=double(cfg.compute.cpu_override_ids(:));
    overrideRates=double(cfg.compute.cpu_override_cycles_per_s(:));
    assert(numel(overrideIDs)==numel(overrideRates), ...
        'SafetyOffloading:CPUOverride','CPU override IDs and rates must have equal length');
    for oi=1:numel(overrideIDs)
        assert(ismember(overrideIDs(oi),ids),'SafetyOffloading:CPUOverride', ...
            'CPU override vehicle %d is not active',overrideIDs(oi));
        s.cpu_rate(overrideIDs(oi))=overrideRates(oi);
    end
end

emptyMeta = struct('valid',false,'packet_id',0,'kind',"",'task_id',0, ...
    'stage',"",'fragment',0,'fragments',0,'src',0,'dst',0,'created_ms',0, ...
    'deadline_ms',0,'seq',0,'flow_id',0,'bytes',0);
s.current = repmat(emptyMeta,maxID,1);
s.empty_meta = emptyMeta;
s.safety_rows = zeros(0,12);
s.radio_rows = cell(0,13);
s.decision_rows = cell(0,20);
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
releaseMs=[]; releaseOrigins=[];
if isfield(cfg.compute,'deterministic_release_ms')
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
        timeMs=0; arrivalIndex=0;
        while true
            arrivalIndex=arrivalIndex+1;
            u=stableUniform(simParams.seed,"task-arrival",origins(o),arrivalIndex);
            timeMs=timeMs+(-log(max(u,realmin))/cfg.compute.arrival_rate_tasks_per_s)*1000;
            if timeMs>=simParams.simulationTime*1000, break; end
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
stationManagement.spv2x = s;
timeManagement.spv2xInitialized = true;
end

