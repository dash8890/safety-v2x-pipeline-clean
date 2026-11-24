function varargout=WorkflowCpuQueue(op,varargin)
%WORKFLOWCPUQUEUE Nonpreemptive EDF CPU with piecewise-constant rate.
% Background jobs compete as real work (no fabricated delay scalars).
switch string(op)
    case "new", varargout{1}=newServer(varargin{:});
    case "enqueue", varargout{1}=enqueue(varargin{:});
    case "settle", [varargout{1},varargout{2}]=settle(varargin{:});
    case "set_rate", [varargout{1},varargout{2}]=setRate(varargin{:});
    case "start_next", varargout{1}=startNext(varargin{:});
    otherwise, error('WorkflowCpuQueue:UnknownOp','Unknown op %s',op);
end
end

function S=newServer(vehicleId,nominalHz,nowS)
S=struct('vehicle_id',vehicleId,'nominal_hz',nominalHz,'available_hz',nominalHz, ...
    'queue',[],'running',[],'last_accounted_s',nowS,'total_served',0, ...
    'completions',{{}});
end

function job=makeJob(jobId,jobType,cycles,deadlineS,enqueueS)
job=struct('job_id',jobId,'job_type',string(jobType),'cycles',cycles, ...
    'remaining',cycles,'deadline_s',deadlineS,'enqueue_s',enqueueS, ...
    'start_s',NaN,'end_s',NaN,'meta',struct());
end

function S=enqueue(S,jobId,jobType,cycles,deadlineS,nowS,meta)
if nargin<7, meta=struct(); end
job=makeJob(jobId,jobType,cycles,deadlineS,nowS);
job.meta=meta;
if isempty(S.queue)
    S.queue=job;
else
    S.queue=[S.queue, job]; %#ok<AGROW>
end
S.queue=sortJobs(S.queue);
if isempty(S.running)
    S=startNext(S,nowS);
end
end

function [S,events]=settle(S,t)
% Accrue service to t; emit completion events at exact crossing times.
events={};
assert(t+1e-15>=S.last_accounted_s,'WorkflowCpuQueue:TimeGoesBack');
while true
    if isempty(S.running)
        S.last_accounted_s=t;
        break
    end
    dt=t-S.last_accounted_s;
    if dt<=0, break; end
    rate=S.available_hz;
    if rate<=0
        S.last_accounted_s=t;
        break
    end
    need=S.running.remaining/rate;
    if need<=dt+1e-15
        doneT=S.last_accounted_s+need;
        served=S.running.remaining;
        S.total_served=S.total_served+served;
        S.running.remaining=0;
        S.running.end_s=doneT;
        events{end+1}=S.running; %#ok<AGROW>
        S.completions{end+1}=S.running; %#ok<AGROW>
        S.running=[];
        S.last_accounted_s=doneT;
        S=startNext(S,doneT);
        if isempty(S.running)
            S.last_accounted_s=t;
            break
        end
    else
        served=rate*dt;
        S.running.remaining=S.running.remaining-served;
        S.total_served=S.total_served+served;
        S.last_accounted_s=t;
        break
    end
end
end

function [S,events]=setRate(S,newHz,t)
[S,events]=settle(S,t);
S.available_hz=newHz;
end

function S=startNext(S,nowS)
if ~isempty(S.running) || isempty(S.queue), return; end
S.queue=sortJobs(S.queue);
S.running=S.queue(1);
S.queue(1)=[];
S.running.start_s=nowS;
end

function q=sortJobs(q)
if numel(q)<=1, return; end
% EDF then enqueue_s then job_id
keys=zeros(numel(q),3);
for i=1:numel(q)
    keys(i,:)=[q(i).deadline_s, q(i).enqueue_s, q(i).job_id];
end
[~,ord]=sortrows(keys,[1 2 3]);
q=q(ord);
end
