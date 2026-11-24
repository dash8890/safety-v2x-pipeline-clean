function runP4ConvergenceProbes()
%RUNP4CONVERGENCEPROBES Serial matched probes for a checkpoint return curve.
root=fileparts(mfilename('fullpath'));
epochs=[20 40 60 80];
for epoch=epochs
    tag=sprintf('epoch%03d',epoch);
    checkpoint=fullfile(root,'drl','outputs','p4_cycle01_checkpoints', ...
        sprintf('fresh_ddqn_%s.json',tag));
    out=fullfile(root,'outputs',sprintf('p4_convergence_cycle01_%s',tag));
    runP4ProbeSuite(out,'workflow_fresh_ddqn','fresh_weights_json',checkpoint);
end
end
