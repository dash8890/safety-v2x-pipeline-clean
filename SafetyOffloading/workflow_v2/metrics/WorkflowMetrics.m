function varargout=WorkflowMetrics(op,varargin)
%WORKFLOWMETRICS Event log + outcome accounting for workflow_v2.
switch string(op)
    case "empty", varargout{1}=emptyLog();
    case "log", varargout{1}=logEvent(varargin{:});
    case "summary", varargout{1}=summarize(varargin{1});
    otherwise, error('WorkflowMetrics:UnknownOp','Unknown op %s',op);
end
end

function L=emptyLog()
L=struct('t_s',{},'type',{},'workflow_id',{},'stage_id',{},'edge_id',{}, ...
    'vehicle_id',{},'detail',{});
end

function L=logEvent(L,t,type,varargin)
e=struct('t_s',t,'type',string(type),'workflow_id',0,'stage_id',"",'edge_id',0, ...
    'vehicle_id',0,'detail',"");
for i=1:2:numel(varargin)
    e.(varargin{i})=varargin{i+1};
end
L(end+1)=e; %#ok<AGROW>
end

function s=summarize(W)
s=struct();
s.n_events=numel(W.log);
s.workflows=struct('workflow_id',{},'status',{},'completion_s',{},'deadline_s',{});
for i=1:numel(W.workflows)
    w=W.workflows(i);
    s.workflows(end+1)=struct('workflow_id',w.workflow_id,'status',w.status, ...
        'completion_s',w.completion_s,'deadline_s',w.absolute_deadline_s); %#ok<AGROW>
end
s.radio_data_fragments=0;
s.radio_ctrl_fragments=0;
s.cpu_served=0;
for v=1:numel(W.cpu)
    s.cpu_served=s.cpu_served+W.cpu(v).total_served;
end
for i=1:numel(W.log)
    if W.log(i).type=="FRAGMENT_TX" || W.log(i).type=="FRAGMENT_RX"
        if contains(string(W.log(i).detail),"ctrl"), s.radio_ctrl_fragments=s.radio_ctrl_fragments+1;
        else, s.radio_data_fragments=s.radio_data_fragments+1; end
    end
end
end
