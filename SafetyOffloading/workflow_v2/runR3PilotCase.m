function summary=runR3PilotCase(caseRoot,policy,seed,varargin)
%RUNR3PILOTCASE One matched R3 world through WiLab Mode-2 (or abstract twin).
% summary = runR3PilotCase(caseRoot, 'workflow_b0_local', 1)
% summary = runR3PilotCase(caseRoot, 'workflow_b4_joint', 1, 'radio_mode', 'abstract_link')
arguments
    caseRoot (1,1) string
    policy (1,1) string
    seed (1,1) double
end
arguments (Repeating)
    varargin
end
radioMode="wilab_mode2";
loadRho=NaN;
baseCfgOverride="";
simTimeOverride=NaN;
freshEps=NaN;
freshExplorationMode="";
freshCoverageRate=NaN;
for i=1:2:numel(varargin)
    switch string(varargin{i})
        case "radio_mode", radioMode=string(varargin{i+1});
        case "cpu_offered_load", loadRho=double(varargin{i+1});
        case "config", baseCfgOverride=string(varargin{i+1});
        case "simulation_time", simTimeOverride=double(varargin{i+1});
        case "fresh_ddqn_epsilon", freshEps=double(varargin{i+1});
        case "fresh_exploration_mode", freshExplorationMode=string(varargin{i+1});
        case "fresh_coverage_rate", freshCoverageRate=double(varargin{i+1});
    end
end

root=fileparts(mfilename('fullpath')); % workflow_v2
safetyRoot=fileparts(root);
repoRoot=fileparts(safetyRoot);
addpath(genpath(repoRoot));
addpath(genpath(fullfile(safetyRoot,'workflow_v2')));

if ~isfolder(caseRoot), mkdir(caseRoot); end
marker=fullfile(caseRoot,'COMPLETE.mat');
if isfile(marker)
    summary=jsondecode(fileread(fullfile(caseRoot,'workflow_v2','summary.json')));
    return
end

if strlength(baseCfgOverride)>0
    baseCfg=baseCfgOverride;
else
    baseCfg=fullfile(root,'config','r3_pilot.json');
end
cfg=jsondecode(fileread(baseCfg));
cfg.workflow.radio_mode=char(radioMode);
if ~isnan(loadRho), cfg.workflow.cpu_offered_load=loadRho; end
if ~isnan(freshEps), cfg.workflow.fresh_ddqn_epsilon=freshEps; end
if strlength(freshExplorationMode)>0, cfg.workflow.fresh_exploration_mode=char(freshExplorationMode); end
if ~isnan(freshCoverageRate), cfg.workflow.fresh_coverage_rate=freshCoverageRate; end
cfgPath=fullfile(caseRoot,'config.json');
fid=fopen(cfgPath,'w'); fprintf(fid,'%s',jsonencode(cfg,'PrettyPrint',true)); fclose(fid);

cfgFile=fullfile(repoRoot,'ConfigFiles','mode2_pipeline.cfg');
simTime=1.0;
if isfield(cfg,'experiment') && isfield(cfg.experiment,'duration_s')
    simTime=double(cfg.experiment.duration_s);
end
if ~isnan(simTimeOverride), simTime=simTimeOverride; end
rho=20;
if isfield(cfg.mobility,'density_vehicles_per_km')
    rho=double(cfg.mobility.density_vehicles_per_km);
    if ~isscalar(rho), rho=rho(1); end
end

common={cfgFile,'Technology','NR-V2X','seed',seed,'simulationTime',simTime, ...
    'rho',rho,'roadLength',1000,'vMean',70,'vStDev',7, ...
    'BwMHz',20,'SCS_NR',30,'MCS_NR',12,'sizeSubchannel',10, ...
    'beaconSizeBytes',1180,'generationInterval',0.1,'allocationPeriod',0.1, ...
    'T1autonomousMode',0.5,'T2autonomousMode',20,'TsensingPeriod',0.1, ...
    'ratioSelectedAutonomousMode',0.2,'powerThresholdAutonomous',-126, ...
    'probResKeep',0.8,'cv2xNumberOfReplicasMax',1,'Raw',150, ...
    'dcc_active',false,'cbrActive',true,'printCBR',false, ...
    'safetyOffloadingEnabled',true,'safetyOffloadingConfig',cfgPath, ...
    'safetyOffloadingPolicy',char(policy),'safetyOffloadingWorld',char(policy), ...
    'outputFolder',char(caseRoot)};

WiLabV2Xsim_final_policy_90(common{:});

sumPath=fullfile(caseRoot,'workflow_v2','summary.json');
if isfile(sumPath)
    summary=jsondecode(fileread(sumPath));
else
    summary=struct('policy',char(policy),'seed',seed,'n_success',0,'n_workflows',0, ...
        'error','missing_summary');
end
completedAt=datetime('now');
save(marker,'summary','completedAt','policy','seed','radioMode');
end
