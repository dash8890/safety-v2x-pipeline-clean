function varargout=WorkflowObservation(op,varargin)
%WORKFLOWOBSERVATION Deployable strip for workflow_v2 (R1-07).
% True helper CPU queues never enter policy input. Helpers appear only via
% received, timestamped HELPER_ADV summaries (or local ego state).
switch string(op)
    case "empty"
        varargout{1}=emptyObs();
    case "strip_deployable"
        varargout{1}=stripDeployable(varargin{1});
    case "ingest_adv"
        varargout{1}=ingestAdv(varargin{:});
    case "assert_no_true_queue"
        varargout{1}=assertNoTrueQueue(varargin{1});
    otherwise
        error('WorkflowObservation:UnknownOp','Unknown op %s',op);
end
end

function o=emptyObs()
o=struct('now_ms',0,'ego_id',0, ...
    'helpers',struct('helper_id',{},'available_cpu_hz',{},'queued_cycles',{}, ...
    'running_remaining_cycles',{},'admitted_pending_cycles',{},'willingness',{}, ...
    'rx_ms',{},'ttl_ms',{},'fresh',{},'source',{}), ...
    'true_helper_cpu_queue',[], ... % diagnostic only; stripped before policy
    'future_contact_ms',[], ...
    'contact_observed',struct('helper_id',{},'last_rx_ms',{},'reachable',{}));
end

function dep=stripDeployable(obs)
dep=obs;
dep.true_helper_cpu_queue=[];
dep.future_contact_ms=[];
if isfield(dep,'helpers')
    for i=1:numel(dep.helpers)
        dep.helpers(i).source="received_adv";
        % Never carry privileged residual fields under another name.
        if isfield(dep.helpers,'true_queued_cycles')
            dep.helpers(i).true_queued_cycles=[]; %#ok<NASGU>
        end
    end
end
% Drop any helper that was never received as ADV (oracle injection).
if isfield(dep,'helpers') && ~isempty(dep.helpers)
    keep=false(1,numel(dep.helpers));
    for i=1:numel(dep.helpers)
        keep(i)=isfield(dep.helpers(i),'rx_ms') && isfinite(dep.helpers(i).rx_ms) ...
            && string(dep.helpers(i).source)~="oracle_true_queue";
    end
    dep.helpers=dep.helpers(keep);
end
end

function obs=ingestAdv(obs,adv,rxMs)
% Policy-visible helper state arrives only through received ADV payloads.
h=struct('helper_id',adv.helper_id,'available_cpu_hz',adv.available_cpu_hz, ...
    'queued_cycles',adv.queued_cycles,'running_remaining_cycles',adv.running_remaining_cycles, ...
    'admitted_pending_cycles',adv.admitted_pending_cycles,'willingness',adv.willingness, ...
    'rx_ms',rxMs,'ttl_ms',adv.ttl_ms,'fresh',true,'source',"received_adv");
found=false;
for i=1:numel(obs.helpers)
    if obs.helpers(i).helper_id==adv.helper_id
        obs.helpers(i)=h; found=true; break
    end
end
if ~found
    obs.helpers(end+1)=h; %#ok<AGROW>
end
c=struct('helper_id',adv.helper_id,'last_rx_ms',rxMs,'reachable',true);
found=false;
for i=1:numel(obs.contact_observed)
    if obs.contact_observed(i).helper_id==adv.helper_id
        obs.contact_observed(i)=c; found=true; break
    end
end
if ~found
    obs.contact_observed(end+1)=c; %#ok<AGROW>
end
end

function tf=assertNoTrueQueue(obs)
tf=true;
if isfield(obs,'true_helper_cpu_queue') && ~isempty(obs.true_helper_cpu_queue)
    tf=false; return
end
if isfield(obs,'future_contact_ms') && ~isempty(obs.future_contact_ms)
    tf=false; return
end
if isfield(obs,'helpers')
    for i=1:numel(obs.helpers)
        if isfield(obs.helpers(i),'source') && obs.helpers(i).source=="oracle_true_queue"
            tf=false; return
        end
    end
end
end
