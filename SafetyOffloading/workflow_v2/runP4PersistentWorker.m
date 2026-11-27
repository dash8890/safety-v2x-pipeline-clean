function runP4PersistentWorker(runName,workerRoot)
%RUNP4PERSISTENTWORKER Serve isolated rollout jobs without MATLAB restart.
arguments
    runName (1,1) string
    workerRoot (1,1) string
end
root=fileparts(mfilename('fullpath'));
if ~isfolder(workerRoot), mkdir(workerRoot); end
fclose(fopen(fullfile(workerRoot,'READY'),'w'));
jobFile=fullfile(workerRoot,'job.json');
stopFile=fullfile(workerRoot,'STOP');
while ~isfile(stopFile)
    if ~isfile(jobFile), pause(0.10); continue; end
    runningFile=fullfile(workerRoot,'running.json');
    [ok,msg]=movefile(jobFile,runningFile,'f');
    if ~ok, warning('%s',msg); pause(0.05); continue; end
    job=jsondecode(fileread(runningFile));
    if isfield(job,'kind') && string(job.kind)=="validation"
        tag=sprintf('validation_%02d',job.block);
    else
        tag=sprintf('%03d',job.rollout_id);
    end
    doneFile=fullfile(workerRoot,"done_"+tag+".json");
    failedFile=fullfile(workerRoot,"failed_"+tag+".json");
    try
        if isfield(job,'kind') && string(job.kind)=="validation"
            runP4OnlineBlockProbes(job.block,runName);
        else
            runP4Online80Episode(job.rollout_id,string(job.weights_json),runName,job.seed_base);
        end
        fid=fopen(doneFile,'w'); fprintf(fid,'%s',jsonencode(struct('status','ok','tag',tag))); fclose(fid);
    catch err
        fid=fopen(failedFile,'w'); fprintf(fid,'%s',jsonencode(struct('status','failed','tag',tag,'message',err.message))); fclose(fid);
    end
    if isfile(runningFile), delete(runningFile); end
    % Explicit memory cleanup between jobs to prevent workspace accumulation.
    clearvars -except runName workerRoot jobFile stopFile
    fclose('all');
end
end
