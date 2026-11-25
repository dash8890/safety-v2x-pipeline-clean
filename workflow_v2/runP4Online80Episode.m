function summary=runP4Online80Episode(rolloutId,weightsJson,runName,seedBase)
%RUNP4ONLINE80EPISODE One frozen-protocol physical online-training rollout.
arguments
    rolloutId (1,1) double {mustBeInteger,mustBePositive,mustBeLessThanOrEqual(rolloutId,88)}
    weightsJson (1,1) string
    runName (1,1) string = "p4_online80_dueling_run01"
    seedBase (1,1) double = 3000
end
root=fileparts(mfilename('fullpath'));
caseRoot=fullfile(root,'outputs',runName,'rollouts',sprintf('rollout%03d',rolloutId));
summary=runR3PilotCase(caseRoot,'workflow_fresh_ddqn',seedBase+rolloutId, ...
    'config',fullfile(root,'config','p3_bc_selected.json'),'cpu_offered_load',0.85, ...
    'fresh_ddqn_epsilon',0.15,'fresh_exploration_mode','coverage', ...
    'fresh_coverage_rate',0.55,'fresh_weights_json',weightsJson);
end
