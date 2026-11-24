function runP4ConvergenceProbes(cycleName)
%RUNP4CONVERGENCEPROBES Serial matched probes for a checkpoint return curve.
root=fileparts(mfilename('fullpath'));
if nargin<1, cycleName="cycle01"; end
epochs=[20 40 60 80];
for epoch=epochs
    tag=sprintf('epoch%03d',epoch);
    checkpoint=fullfile(root,'drl','outputs',sprintf('p4_%s_checkpoints',cycleName), ...
        sprintf('fresh_ddqn_%s.json',tag));
    out=fullfile(root,'outputs',sprintf('p4_convergence_%s_%s',cycleName,tag));
    runP4ProbeSuite(out,'workflow_fresh_ddqn','fresh_weights_json',checkpoint);
end
end
