function summary=runP4Online80Episode(episodeId,weightsJson)
%RUNP4ONLINE80EPISODE One frozen-protocol physical online-training rollout.
arguments
    episodeId (1,1) double {mustBeInteger,mustBePositive,mustBeLessThanOrEqual(episodeId,80)}
    weightsJson (1,1) string
end
root=fileparts(mfilename('fullpath'));
caseRoot=fullfile(root,'outputs','p4_online80','episodes',sprintf('episode%03d',episodeId));
summary=runR3PilotCase(caseRoot,'workflow_fresh_ddqn',3000+episodeId, ...
    'config',fullfile(root,'config','p3_bc_selected.json'),'cpu_offered_load',0.85, ...
    'fresh_ddqn_epsilon',0.15,'fresh_exploration_mode','coverage', ...
    'fresh_coverage_rate',0.55,'fresh_weights_json',weightsJson);
end
