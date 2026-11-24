function out=runBaselineAcceptance()
%RUNBASELINEACCEPTANCE Fresh action-schema gates before comparator/RL runs.
% This is intentionally small: R1/R2 remain the full transport/CPU suite;
% these checks ensure every controller shares the fresh legal action API.
here=fileparts(mfilename('fullpath')); root=fileparts(here); addpath(genpath(root));
r2=runR2Acceptance();
checks={@testFreshActionUnion,@testAckGatedRelease,@testNoFictitiousTime};
rows=struct('id',{},'ok',{},'msg',{});
for i=1:numel(checks)
    try
        msg=checks{i}(); rows(end+1)=struct('id',sprintf('B0-%02d',i),'ok',true,'msg',string(msg)); %#ok<AGROW>
        fprintf('[PASS] %s\n',msg);
    catch ME
        rows(end+1)=struct('id',sprintf('B0-%02d',i),'ok',false,'msg',string(ME.message)); %#ok<AGROW>
        fprintf('[FAIL] %s\n',ME.message);
    end
end
out=struct('r2_pass',r2.all_pass,'rows',rows,'all_pass',r2.all_pass && all([rows.ok]));
assert(out.all_pass,'runBaselineAcceptance:Failed','Fresh action acceptance failed');
end

function msg=testFreshActionUnion()
W=WorkflowEngine('new',struct('n_vehicles',3)); [W,~]=WorkflowEngine('create_workflow',W,0);
c=WorkflowCandidateBuilder(W); kinds=string({c.kind});
assert(any(kinds=="COMMIT_LOCAL") && any(kinds=="REQUEST_REMOTE_COMMIT"));
assert(any(kinds=="REJECT_WORKFLOW") && any(kinds=="STOP_PLAN"));
assert(~any(kinds=="DROP") && ~any(kinds=="REQUEST_REMOTE"));
msg='fresh action union contains only executable contract names';
end

function msg=testAckGatedRelease()
W=WorkflowEngine('new',struct('n_vehicles',3)); [W,id]=WorkflowEngine('create_workflow',W,0);
W=WorkflowActionValidator('apply',W,findAction(WorkflowCandidateBuilder(W),"COMMIT_LOCAL",id,"A",0));
W=WorkflowEngine('advance_to',W,0.020);
a=findAction(WorkflowCandidateBuilder(W),"REQUEST_REMOTE_COMMIT",id,"B",2);
W=WorkflowActionValidator('apply',W,a);
c=WorkflowCandidateBuilder(W); kinds=string({c.kind});
assert(~any(kinds=="RELEASE_EDGE"),'release must remain ACK-gated');
msg='remote release absent until COMMIT_ACK';
end

function msg=testNoFictitiousTime()
W=WorkflowEngine('new',struct('n_vehicles',3)); [W,~]=WorkflowEngine('create_workflow',W,0); t=W.now_s;
c=WorkflowCandidateBuilder(W); a=findAction(c,"STOP_PLAN",0,"",0);
W=WorkflowActionValidator('apply',W,a); assert(W.now_s==t);
msg='STOP_PLAN has no fictitious physical-time progression';
end

function a=findAction(c,kind,wf,stage,helper)
for i=1:numel(c)
    if c(i).kind==string(kind) && c(i).workflow_id==wf && c(i).stage_id==string(stage) && c(i).helper_id==helper
        a=c(i); return
    end
end
error('runBaselineAcceptance:Missing','required action not found');
end
