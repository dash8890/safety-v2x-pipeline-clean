function varargout=WorkflowActionValidator(op,varargin)
%WORKFLOWACTIONVALIDATOR Validate/apply one fresh-schema action atomically.
switch string(op)
    case "validate", varargout{1}=validate(varargin{1},varargin{2});
    case "apply", varargout{1}=apply(varargin{1},varargin{2});
    case "legacy_to_fresh", varargout{1}=legacyToFresh(varargin{1},varargin{2});
    otherwise, error('WorkflowActionValidator:UnknownOp','Unknown op %s',op);
end
end
function ok=validate(W,a)
c=WorkflowCandidateBuilder(W); ok=false;
for i=1:numel(c)
    x=c(i);
    if x.kind==string(a.kind) && x.workflow_id==a.workflow_id && x.stage_id==string(a.stage_id) && ...
            x.helper_id==a.helper_id && x.edge_id==a.edge_id, ok=true; return; end
end
end
function W=apply(W,a)
assert(validate(W,a),'WorkflowActionValidator:Illegal','Illegal action %s',string(a.kind));
switch string(a.kind)
    case "COMMIT_LOCAL", W=WorkflowEngine('commit_local',W,a.workflow_id,a.stage_id);
    case "REQUEST_REMOTE_COMMIT", W=WorkflowEngine('request_remote',W,a.workflow_id,a.stage_id,a.helper_id);
    case "RELEASE_EDGE"
        ed=edgeById(W,a.workflow_id,a.edge_id); W=WorkflowEngine('release_edge',W,a.workflow_id,ed.name);
    case "REJECT_WORKFLOW", W=WorkflowEngine('drop_workflow',W,a.workflow_id);
    case "STOP_PLAN"
    otherwise, error('WorkflowActionValidator:Kind','Unsupported action %s',string(a.kind));
end
end
function a=legacyToFresh(W,x)
a=struct('kind',"STOP_PLAN",'workflow_id',0,'stage_id',"",'helper_id',0,'edge_id',0,'detail',string(x.detail));
switch string(x.type)
    case "COMMIT_LOCAL", a.kind="COMMIT_LOCAL"; a.workflow_id=x.workflow_id; a.stage_id=x.stage_id;
    case "REQUEST_REMOTE", a.kind="REQUEST_REMOTE_COMMIT"; a.workflow_id=x.workflow_id; a.stage_id=x.stage_id; a.helper_id=x.helper_id;
    case "RELEASE_EDGE"
        a.kind="RELEASE_EDGE"; a.workflow_id=x.workflow_id;
        ed=edgeByName(W,x.workflow_id,x.edge_name); a.edge_id=ed.edge_id;
    case "DROP", a.kind="REJECT_WORKFLOW"; a.workflow_id=x.workflow_id;
end
end
function ed=edgeById(W,wfId,edgeId)
for i=1:numel(W.edges)
    if W.edges(i).workflow_id==wfId && W.edges(i).edge_id==edgeId, ed=W.edges(i); return; end
end
error('WorkflowActionValidator:Edge','Unknown edge');
end
function ed=edgeByName(W,wfId,name)
for i=1:numel(W.edges)
    if W.edges(i).workflow_id==wfId && W.edges(i).name==string(name), ed=W.edges(i); return; end
end
error('WorkflowActionValidator:Edge','Unknown edge');
end
