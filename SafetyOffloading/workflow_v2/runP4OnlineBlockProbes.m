function runP4OnlineBlockProbes(updateBlock,runName,phase)
%RUNP4ONLINEBLOCKPROBES Fixed five-seed probe every five training epochs.
arguments
    updateBlock (1,1) double {mustBeInteger,mustBePositive}
    runName (1,1) string = "p4_online80_dueling_run01"
    phase (1,1) string = "main"
end
root=fileparts(mfilename('fullpath'));
updateRoot=fullfile(root,'outputs',runName,'training',sprintf('update%02d',updateBlock));
for epoch=5:5:80
    weights=fullfile(updateRoot,'optimizer_checkpoints',sprintf('dueling_ddqn_update%03d.json',epoch));
    out=fullfile(root,'outputs',runName,'probes',phase,sprintf('update%02d',updateBlock),sprintf('update%03d',epoch));
    runP4ProbeSuite(out,'workflow_fresh_ddqn','fresh_weights_json',weights);
end
end
