function varargout=WorkflowEventDispatcher(op,varargin)
%WORKFLOWEVENTDISPATCHER Normative event order (contract §B3) for R2 unit mode.
switch string(op)
    case "step_to", varargout{1}=stepTo(varargin{:});
    otherwise, error('WorkflowEventDispatcher:UnknownOp','Unknown op %s',op);
end
end

function W=stepTo(W,t)
% Process everything up to and including t.
if t+1e-12<W.now_s
    return % already past (WiLab tick / abstract coalescing)
end
% 1. Settle CPU through t (exact completions may be < t)
for v=1:numel(W.cpu)
    [W.cpu(v),evs]=WorkflowCpuQueue('settle',W.cpu(v),t);
    for k=1:numel(evs)
        job=evs{k};
        % Snap time to completion for materialization
        W.now_s=job.end_s;
        W=fevalAt(W,'onCpuComplete',job,v);
        % Mark RUNNING successor if CPU started next
        W=syncRunningFlags(W,v);
    end
end
W.now_s=t;
% 2. Mobility / capacity already applied by callers via set_contact/set_rate before step
% 3. Deliver radio / control due at t
due=false(1,numel(W.event_q));
for i=1:numel(W.event_q)
    due(i)=W.event_q{i}.t_s<=t+1e-15;
end
evs=W.event_q(due);
W.event_q=W.event_q(~due);
% Sort by kind priority within timestamp: CTRL/FRAGMENT before CONTROLLER
if ~isempty(evs)
    pri=zeros(numel(evs),1);
    ts=zeros(numel(evs),1);
    for i=1:numel(evs)
        ts(i)=evs{i}.t_s;
        switch evs{i}.kind
            case "CTRL_DELIVER", pri(i)=1;
            case "FRAGMENT_DELIVER", pri(i)=2;
            case "CONTROLLER", pri(i)=8;
            otherwise, pri(i)=5;
        end
    end
    [~,ord]=sortrows([ts,pri],[1 2]);
    evs=evs(ord);
end
for i=1:numel(evs)
    W.now_s=evs{i}.t_s;
    W=dispatchOne(W,evs{i});
end
W.now_s=t;
% 4. CPU materialization already done in settle loop
% 5. Arrivals handled by create_workflow
% 6. Success check
for i=1:numel(W.workflows)
    W=fevalAt(W,'checkWorkflowSuccess',W.workflows(i).workflow_id);
end
% 7. Expire
W=fevalAt(W,'expireWorkflows',t);
% 8-9. Controller ticks are explicit in tests / CONTROLLER events
end

function W=dispatchOne(W,ev)
switch ev.kind
    case "CTRL_DELIVER"
        p=ev.payload;
        if ~contactOkLocal(W,p.src,p.dst)
            W.log=WorkflowMetrics('log',W.log,W.now_s,"CTRL_DROP_NO_CONTACT", ...
                'detail',string(p.msg.ctrl_type));
            return
        end
        W=fevalAt(W,'handleCtrl',p.msg,p.src,p.dst);
    case "FRAGMENT_DELIVER"
        p=ev.payload;
        if ~contactOkLocal(W,p.src,p.dst)
            W.log=WorkflowMetrics('log',W.log,W.now_s,"FRAGMENT_DROP_NO_CONTACT", ...
                'workflow_id',p.workflow_id,'detail',string(p.edge_name));
            return
        end
        W=fevalAt(W,'handleFragment',p);
    case "CONTROLLER"
        % Tests drive explicit actions; log only
        W.log=WorkflowMetrics('log',W.log,W.now_s,"CONTROLLER_TICK", ...
            'workflow_id',ev.payload.workflow_id);
    otherwise
        W.log=WorkflowMetrics('log',W.log,W.now_s,"UNKNOWN_EVENT",'detail',ev.kind);
end
end

function W=syncRunningFlags(W,v)
S=W.cpu(v);
if isempty(S.running), return; end
if ~isfield(S.running,'meta') || ~isfield(S.running.meta,'stage_id'), return; end
wfId=S.running.meta.workflow_id; sid=S.running.meta.stage_id;
for i=1:numel(W.stages)
    if W.stages(i).workflow_id==wfId && W.stages(i).stage_id==string(sid) ...
            && W.stages(i).status=="QUEUED"
        W.stages(i).status="RUNNING";
        W.stages(i).start_s=S.running.start_s;
        if string(sid)=="D", W.last_d_start_s=S.running.start_s; end
        W.log=WorkflowMetrics('log',W.log,S.running.start_s,"STAGE_START", ...
            'workflow_id',wfId,'stage_id',string(sid),'vehicle_id',v);
    end
end
end

function tf=contactOkLocal(W,a,b)
if a==b, tf=true; return; end
tf=W.contact(a,b);
end

function W=fevalAt(W,name,varargin)
% Call nested-equivalent helpers exposed via engine private pattern:
% re-enter WorkflowEngine internal by duplicating critical calls here through
% a package of engine ops where possible; for private helpers, inline bridges.
switch string(name)
    case "onCpuComplete"
        W=bridgeOnCpuComplete(W,varargin{:});
    case "checkWorkflowSuccess"
        W=bridgeCheckSuccess(W,varargin{:});
    case "expireWorkflows"
        W=bridgeExpire(W,varargin{:});
    case "handleCtrl"
        W=bridgeHandleCtrl(W,varargin{:});
    case "handleFragment"
        W=bridgeHandleFragment(W,varargin{:});
    otherwise
        error('WorkflowEventDispatcher:Bridge','no bridge for %s',name);
end
end

% --- Bridges call back into WorkflowEngine public ops / duplicated minimal logic ---
% onCpuComplete / handle* live inside WorkflowEngine.m as local functions and are
% NOT visible here. Expose them as engine ops.

function W=bridgeOnCpuComplete(W,job,vehicleId)
W=WorkflowEngine('__on_cpu_complete',W,job,vehicleId);
end
function W=bridgeCheckSuccess(W,wfId)
W=WorkflowEngine('__check_success',W,wfId);
end
function W=bridgeExpire(W,t)
W=WorkflowEngine('__expire',W,t);
end
function W=bridgeHandleCtrl(W,msg,src,dst)
W=WorkflowEngine('__handle_ctrl',W,msg,src,dst);
end
function W=bridgeHandleFragment(W,p)
W=WorkflowEngine('__handle_fragment',W,p);
end
