function candidates=WorkflowCandidateBuilder(W,varargin)
%WORKFLOWCANDIDATEBUILDER Build the fresh executable causal action union.
% Candidate construction reads origin-visible state only. Remote actions are
% requests, never fictitious ACKs; physical time never advances here.
opts=struct('include_reject',true);
for i=1:2:numel(varargin), opts.(char(varargin{i}))=varargin{i+1}; end
candidates=emptyActions();
for i=1:numel(W.stages)
    st=W.stages(i);
    if st.status~="ELIGIBLE", continue; end
    candidates(end+1)=action("COMMIT_LOCAL",st.workflow_id,st.stage_id,0,0,"local"); %#ok<AGROW>
    for helper=1:numel(W.cpu)
        if helper==W.origin_id || ~observedReachable(W,helper), continue; end
        candidates(end+1)=action("REQUEST_REMOTE_COMMIT",st.workflow_id,st.stage_id,helper,0,"request_only"); %#ok<AGROW>
    end
end
for i=1:numel(W.edges)
    ed=W.edges(i);
    if ed.status=="READY_HELD" && edgeReleaseLegal(W,ed)
        candidates(end+1)=action("RELEASE_EDGE",ed.workflow_id,"",0,ed.edge_id,char(ed.name)); %#ok<AGROW>
    end
end
if opts.include_reject
    for i=1:numel(W.workflows)
        wf=W.workflows(i);
        if wf.status=="ACTIVE" && ~hasCommitment(W,wf.workflow_id)
            candidates(end+1)=action("REJECT_WORKFLOW",wf.workflow_id,"",0,0,"uncommitted"); %#ok<AGROW>
        end
    end
end
candidates(end+1)=action("STOP_PLAN",0,"",0,0,"no_fictitious_progress");
end

function tf=observedReachable(W,helper)
tf=helper<=size(W.contact,2) && logical(W.contact(W.origin_id,helper));
end
function tf=edgeReleaseLegal(W,ed)
if ed.src_vehicle<=0 || ed.dst_vehicle<=0, tf=false; return; end
if ed.src_vehicle==ed.dst_vehicle, tf=true; return; end
if ~observedReachable(W,ed.dst_vehicle), tf=false; return; end
if strlength(ed.dst_stage)==0, tf=true; return; end
[st,~]=WorkflowEngine('find_stage',W,ed.workflow_id,ed.dst_stage); tf=st.commitment_acked;
end
function tf=hasCommitment(W,wfId)
tf=false;
for i=1:numel(W.stages)
    s=W.stages(i);
    if s.workflow_id==wfId && s.status~="BLOCKED" && s.status~="ELIGIBLE", tf=true; return; end
end
end
function a=action(kind,wf,stage,helper,edge,detail)
a=struct('kind',string(kind),'workflow_id',double(wf),'stage_id',string(stage), ...
    'helper_id',double(helper),'edge_id',double(edge),'detail',string(detail));
end
function a=emptyActions()
a=struct('kind',{},'workflow_id',{},'stage_id',{},'helper_id',{},'edge_id',{},'detail',{});
end
