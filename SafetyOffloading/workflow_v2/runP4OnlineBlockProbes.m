function runP4OnlineBlockProbes(updateBlock)
%RUNP4ONLINEBLOCKPROBES Fixed five-seed probe every five training epochs.
arguments
    updateBlock (1,1) double {mustBeInteger,mustBePositive}
end
root=fileparts(mfilename('fullpath'));
updateRoot=fullfile(root,'outputs','p4_online80','training',sprintf('update%02d',updateBlock));
for epoch=5:5:80
    weights=fullfile(updateRoot,'epoch_checkpoints',sprintf('fresh_ddqn_epoch%03d.json',epoch));
    out=fullfile(root,'outputs','p4_online80','probes',sprintf('update%02d',updateBlock),sprintf('epoch%03d',epoch));
    runP4ProbeSuite(out,'workflow_fresh_ddqn','fresh_weights_json',weights);
end
end
