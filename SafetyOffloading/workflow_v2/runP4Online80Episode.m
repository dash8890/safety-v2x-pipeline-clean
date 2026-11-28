function summary=runP4Online80Episode(rolloutId,weightsJson,runName,seedBase)
%RUNP4ONLINE80EPISODE One frozen-protocol physical online-training rollout.
arguments
    rolloutId (1,1) double {mustBeInteger,mustBePositive,mustBeLessThanOrEqual(rolloutId,88)}
    weightsJson (1,1) string
    runName (1,1) string = "p4_online80_dueling_run01"
    seedBase (1,1) double = 3000
end
% Anneal coverage rate and epsilon linearly from rollout 1 to rollout 60,
% then hold at floor. Early rollouts need state diversity; later rollouts
% should reflect the learned policy so the replay buffer shifts on-policy.
annealEnd = 60;
t = min(1.0, (rolloutId - 1) / (annealEnd - 1));
coverageRate = 0.55 * (1 - t);
epsilon      = 0.15 - 0.10 * t;   % 0.15 -> 0.05
root=fileparts(mfilename('fullpath'));
caseRoot=fullfile(root,'outputs',runName,'rollouts',sprintf('rollout%03d',rolloutId));
summary=runR3PilotCase(caseRoot,'workflow_fresh_ddqn',seedBase+rolloutId, ...
    'config',fullfile(root,'config','p3_bc_selected.json'),'cpu_offered_load',0.85, ...
    'fresh_ddqn_epsilon',epsilon,'fresh_exploration_mode','coverage', ...
    'fresh_coverage_rate',coverageRate,'fresh_weights_json',weightsJson);
end
