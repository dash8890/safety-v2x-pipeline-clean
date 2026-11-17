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
s.next_tick_s = 0;
s.tick_s = phyParams.TTI;
s.packet_counter = 0;
s.evaluation_range_m = min(cfg.safety.receiver_range_m,max(phyParams.Raw));
s.safety_seq = zeros(maxID,1);
s.pending_tx = cell(maxID,1);
s.cpu_queue = cell(maxID,1);
s.cpu_active = zeros(maxID,1);
s.cpu_finish_s = inf(maxID,1);
s.cpu_rate = zeros(maxID,1);
for v = ids'
    u = stableUniform(simParams.seed,"wilab-cpu",v);
    s.cpu_rate(v) = cfg.compute.cpu_cycles_per_s_min + ...
        u*(cfg.compute.cpu_cycles_per_s_max-cfg.compute.cpu_cycles_per_s_min);
end

emptyMeta = struct('valid',false,'packet_id',0,'kind',"",'task_id',0, ...
    'stage',"",'fragment',0,'fragments',0,'src',0,'dst',0,'created_ms',0, ...
    'deadline_ms',0,'seq',0,'flow_id',0,'bytes',0);
s.current = repmat(emptyMeta,maxID,1);
s.empty_meta = emptyMeta;
s.safety_rows = zeros(0,12);
s.radio_rows = cell(0,13);
s.decision_rows = cell(0,19);
s.output_folder = string(outParams.outputFolder);

originCount = max(1,round(numel(ids)*cfg.compute.origin_fraction));
originIndex = unique(round(linspace(1,numel(ids),originCount)));
origins = ids(originIndex);
releaseMs = (0:cfg.compute.arrival_interval_ms:(simParams.simulationTime*1000-1))';
nTasks = numel(releaseMs)*numel(origins);
taskTemplate = struct('task_id',0,'origin',0,'release_ms',0,'deadline_ms',0, ...
    'input_bytes',0,'output_bytes',0,'cycles',0,'value',1,'action',"pending", ...
    'helper',0,'status',"pending",'input_fragments',0,'input_received',0, ...
    'output_fragments',0,'output_received',0,'cpu_owner',0,'cpu_start_ms',NaN, ...
    'cpu_finish_ms',NaN,'input_delivered_ms',NaN,'result_delivered_ms',NaN, ...
    'decision_id',0);
s.tasks = repmat(taskTemplate,nTasks,1);
k = 0;
for r = 1:numel(releaseMs)
    for o = 1:numel(origins)
        k=k+1; t=taskTemplate; t.task_id=k; t.origin=origins(o); t.release_ms=releaseMs(r);
        t.deadline_ms=releaseMs(r)+cfg.compute.deadline_ms; t.input_bytes=cfg.compute.input_bytes;
        t.output_bytes=cfg.compute.output_bytes; t.cycles=cfg.compute.cycles_per_task;
        s.tasks(k)=t;
    end
end
s.release_cursor = 1;
s.decision_counter = 0;
s.scenario_id = simParams.seed;
s.position_snapshot = positionManagement;
stationManagement.spv2x = s;
timeManagement.spv2xInitialized = true;
end

