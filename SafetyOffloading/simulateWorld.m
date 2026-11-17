function result = simulateWorld(cfg, trace, policy)
%SIMULATEWORLD Packet-level NR-V2X Mode-2 safety/offloading simulation.
arguments
    cfg struct
    trace struct
    policy (1,1) string
end

slot_ms = cfg.radio.slot_duration_ms;
nSlots = floor(cfg.experiment.duration_s * 1000 / slot_ms);
N = numel(trace.vehicle_id);
selectionSlots = max(1, round(cfg.radio.selection_window_ms / slot_ms));
rriSlots = max(1, round(cfg.radio.reservation_interval_ms / slot_ms));
tb = cfg.radio.max_transport_block_bytes;

emptyPacket = struct('id',0,'traffic_class',"",'src',0,'dst',0,'created_ms',0, ...
    'deadline_ms',0,'bytes',0,'task_id',0,'stage',"",'fragment',0, ...
    'fragments',0,'flow_id',0,'seq',0,'rx_ids',[],'scheduled_slot',0, ...
    'subchannel',0,'priority',0);
packets = repmat(emptyPacket, 0, 1);
safetyPackets = repmat(emptyPacket, 0, 1);
nextPacketId = 0;
occupiedHistory = false(max(1,round(cfg.radio.sensing_window_ms/slot_ms)), cfg.radio.subchannels);
historyCursor = 1;

taskCount = height(trace.tasks);
task = repmat(struct('action',"pending",'helper',0,'status',"pending", ...
    'input_received',0,'input_fragments',0,'output_received',0,'output_fragments',0, ...
    'cpu_owner',0,'cpu_remaining',0,'cpu_start_ms',NaN,'cpu_finish_ms',NaN, ...
    'input_delivered_ms',NaN,'result_delivered_ms',NaN,'decision_id',0), taskCount, 1);
cpuQueue = cell(N,1);
cpuActive = zeros(N,1);

% Numeric logs are converted to named tables at the end.
safetyRows = zeros(0,11); % flow seq tx rx created deadline decoded success bytes range bin
decisionRows = cell(0,10);
radioRows = cell(0,13);
occupancy = false(nSlots, cfg.radio.subchannels);
releaseCursor = 1;
decisionId = 0;

for slot = 1:nSlots
    now_ms = (slot-1) * slot_ms;
    [x,y] = vehiclePositions(trace, cfg, now_ms);

    % Complete or advance non-preemptive EDF processors.
    for v = 1:N
        if cpuActive(v) == 0 && ~isempty(cpuQueue{v})
            ids = cpuQueue{v}; deadlines = trace.tasks.deadline_ms(ids);
            [~,ord] = sortrows([deadlines(:), ids(:)], [1 2]);
            tid = ids(ord(1)); cpuQueue{v}(ord(1)) = [];
            if now_ms < trace.tasks.deadline_ms(tid)
                cpuActive(v) = tid; task(tid).cpu_start_ms = now_ms;
                task(tid).cpu_remaining = trace.tasks.cycles(tid);
            else
                task(tid).status = "expired_cpu_queue";
            end
        end
        tid = cpuActive(v);
        if tid > 0
            task(tid).cpu_remaining = task(tid).cpu_remaining - trace.cpu_rate(v)*(slot_ms/1000);
            if task(tid).cpu_remaining <= 0
                task(tid).cpu_finish_ms = now_ms + slot_ms;
                cpuActive(v) = 0;
                if task(tid).action == "local"
                    if task(tid).cpu_finish_ms <= trace.tasks.deadline_ms(tid)
                        task(tid).status = "completed";
                        task(tid).result_delivered_ms = task(tid).cpu_finish_ms;
                    else
                        task(tid).status = "late_cpu";
                    end
                else
                    [newPackets,nextPacketId] = makeComputePackets(nextPacketId, tid, v, ...
                        trace.tasks.origin_id(tid), trace.tasks.output_bytes(tid), "result", ...
                        now_ms+slot_ms, trace.tasks.deadline_ms(tid), tb, slot, selectionSlots, ...
                        cfg.radio.subchannels, trace.seed, policy);
                    task(tid).output_fragments = numel(newPackets);
                    packets = [packets; newPackets]; %#ok<AGROW>
                end
            end
        end
    end

    % Release computation tasks and apply the selected admission policy.
    while releaseCursor <= taskCount && trace.tasks.release_ms(releaseCursor) <= now_ms + 1e-9
        tid = releaseCursor; origin = trace.tasks.origin_id(tid);
        cbr = mean(occupiedHistory, 'all');
        candidates = helperCandidates(origin, x, y, cfg.compute.helper_range_m, cfg.compute.max_helpers);
        [action,helper,riskUpper] = chooseAction(policy, tid, origin, candidates, ...
            cpuQueue, cpuActive, trace, cfg, cbr, x, y);
        decisionId = decisionId + 1;
        task(tid).action = action; task(tid).helper = helper; task(tid).decision_id = decisionId;
        mask = sprintf("reject=1;defer=0;local=1;helpers=%s", mat2str(candidates));
        decisionRows(end+1,:) = {trace.seed,decisionId,now_ms,origin,tid,mask,char(action),riskUpper,char(policy),0}; %#ok<AGROW>
        if action == "local"
            if numel(cpuQueue{origin}) < cfg.compute.queue_capacity
                cpuQueue{origin}(end+1) = tid;
                task(tid).cpu_owner = origin; task(tid).status = "queued_local";
            else
                task(tid).status = "rejected_cpu_capacity";
            end
        elseif action == "offload"
            [newPackets,nextPacketId] = makeComputePackets(nextPacketId, tid, origin, helper, ...
                trace.tasks.input_bytes(tid), "input", now_ms, trace.tasks.deadline_ms(tid), ...
                tb, slot, selectionSlots, cfg.radio.subchannels, trace.seed, policy);
            task(tid).input_fragments = numel(newPackets);
            task(tid).cpu_owner = helper; task(tid).status = "input_in_flight";
            packets = [packets; newPackets]; %#ok<AGROW>
        else
            task(tid).status = "rejected";
        end
        releaseCursor = releaseCursor + 1;
    end

    % Periodic independent safety generation with persistent resource choices.
    safetyPeriodSlots = max(1, round(cfg.safety.generation_interval_ms/slot_ms));
    if mod(slot-1, safetyPeriodSlots) == 0
        seq = floor((slot-1)/safetyPeriodSlots) + 1;
        for src = 1:N
            dx = wrappedDistance(x, x(src), cfg.mobility.road_length_m);
            d = hypot(dx, y-y(src));
            rx = find(d > 0 & d <= cfg.safety.receiver_range_m)';
            if isempty(rx), continue; end
            created = now_ms;
            deadline = created + cfg.safety.deadline_ms;
            flow = src;
            bins = discretize(d(rx), cfg.safety.distance_bins_m, 'IncludedEdge','right');
            rowStart = size(safetyRows,1) + 1;
            for q = 1:numel(rx)
                safetyRows(end+1,:) = [flow seq src rx(q) created deadline NaN 0 ...
                    cfg.safety.packet_bytes d(rx(q)) bins(q)]; %#ok<AGROW>
            end
            nextPacketId = nextPacketId + 1;
            subch = 1 + floor(stableUniform(trace.seed,"sps-subch",src) * cfg.radio.subchannels);
            offset = floor(stableUniform(trace.seed,"sps-offset",src) * min(rriSlots,selectionSlots));
            p = emptyPacket; p.id=nextPacketId; p.traffic_class="safety"; p.src=src; p.dst=0;
            p.created_ms=created; p.deadline_ms=deadline; p.bytes=cfg.safety.packet_bytes;
            p.flow_id=flow; p.seq=seq; p.rx_ids=rx; p.scheduled_slot=min(nSlots,slot+offset);
            p.subchannel=subch; p.priority=1; p.stage=string(rowStart); p.fragments=1; p.fragment=1;
            safetyPackets(end+1,1) = p; %#ok<AGROW>
        end
    end

    % Expire queued radio packets before transmission.
    if ~isempty(packets)
        expired = [packets.deadline_ms] < now_ms - 1e-9;
        for pi = find(expired)
            if packets(pi).task_id > 0 && task(packets(pi).task_id).status ~= "completed"
                task(packets(pi).task_id).status = "radio_expired";
            end
        end
        packets(expired) = [];
    end

    if ~isempty(safetyPackets)
        expiredSafety = [safetyPackets.deadline_ms] < now_ms - 1e-9;
        safetyPackets(expiredSafety) = [];
    end

    due = find([packets.scheduled_slot] == slot);
    dueSafety = find([safetyPackets.scheduled_slot] == slot);
    if ~isempty(due) || ~isempty(dueSafety)
        txPackets = [safetyPackets(dueSafety); packets(due)];
        % Priority-only control drops colliding compute submissions before PHY.
        if policy == "priority_only"
            classes = string([txPackets.traffic_class]);
            safetyChannels = [txPackets(classes=="safety").subchannel];
            keep = true(size(txPackets));
            for qi = 1:numel(txPackets)
                isCompute = any(string(txPackets(qi).traffic_class) == "compute",'all');
                if isCompute && ismember(txPackets(qi).subchannel,safetyChannels)
                    keep(qi) = false;
                    task(txPackets(qi).task_id).status = "preempted_by_safety";
                end
            end
            txPackets = txPackets(keep);
        end
        transmitting = unique([txPackets.src]);
        for qi = 1:numel(txPackets)
            p = txPackets(qi); occupancy(slot,p.subchannel) = true;
            if string(p.traffic_class) == "safety", receivers = p.rx_ids; else, receivers = p.dst; end
            for rxIndex = 1:numel(receivers)
                rx = receivers(rxIndex);
                dx = wrappedDistance(x(rx),x(p.src),cfg.mobility.road_length_m);
                dist = max(1,hypot(dx,y(rx)-y(p.src)));
                halfDuplexLoss = cfg.radio.half_duplex && ismember(rx,transmitting);
                signalDbm = cfg.radio.tx_power_dbm + 2*cfg.radio.antenna_gain_dbi - pathLossDb(dist,cfg.radio.carrier_frequency_ghz);
                interferenceMw = 0;
                interferers = txPackets([txPackets.subchannel] == p.subchannel & [txPackets.src] ~= p.src);
                for jj = 1:numel(interferers)
                    dji = max(1,hypot(wrappedDistance(x(rx),x(interferers(jj).src),cfg.mobility.road_length_m),y(rx)-y(interferers(jj).src)));
                    interferenceMw = interferenceMw + db2powLocal(cfg.radio.tx_power_dbm + ...
                        2*cfg.radio.antenna_gain_dbi - pathLossDb(dji,cfg.radio.carrier_frequency_ghz));
                end
                noiseDbm = -174 + 10*log10((cfg.radio.bandwidth_mhz*1e6)/cfg.radio.subchannels) + cfg.radio.noise_figure_db;
                sinrDb = signalDbm - 10*log10(db2powLocal(noiseDbm)+interferenceMw);
                % A receiver may see several simultaneous replicas; use the
                % conservative effective SINR for this packet opportunity.
                sinrDb = min(sinrDb(:));
                per = min(1,max(0,1./(1+exp(1.15*(sinrDb-4.0)))));
                if string(p.traffic_class) == "safety"
                    decodeUniform = stableUniform(trace.seed,"safety-decode",p.flow_id,p.seq,rx);
                else
                    decodeUniform = stableUniform(trace.seed,"compute-decode",p.task_id,p.stage,p.fragment,rx);
                end
                success = all(~halfDuplexLoss(:)) && all(decodeUniform(:) > per);
                radioRows(end+1,:) = {trace.seed,now_ms,p.src,p.id,char(string(p.traffic_class)),p.priority,p.subchannel,p.seq,char(p.stage),sinrDb,1,success,rx}; %#ok<AGROW>
                if string(p.traffic_class) == "safety"
                    rows = find(safetyRows(:,1)==p.flow_id & safetyRows(:,2)==p.seq & safetyRows(:,4)==rx);
                    if success && ~isempty(rows)
                        safetyRows(rows(1),7) = now_ms+slot_ms;
                        safetyRows(rows(1),8) = (now_ms+slot_ms <= p.deadline_ms);
                    end
                elseif success
                    tid = p.task_id;
                    if p.stage == "input"
                        task(tid).input_received = task(tid).input_received + 1;
                        if task(tid).input_received == task(tid).input_fragments
                            task(tid).input_delivered_ms = now_ms+slot_ms;
                            if numel(cpuQueue{p.dst}) < cfg.compute.queue_capacity
                                cpuQueue{p.dst}(end+1)=tid; task(tid).status="queued_remote";
                            else
                                task(tid).status="rejected_helper_capacity";
                            end
                        end
                    else
                        task(tid).output_received = task(tid).output_received + 1;
                        if task(tid).output_received == task(tid).output_fragments
                            task(tid).result_delivered_ms = now_ms+slot_ms;
                            if task(tid).result_delivered_ms <= trace.tasks.deadline_ms(tid)
                                task(tid).status="completed";
                            else
                                task(tid).status="late_result";
                            end
                        end
                    end
                end
            end
        end
        packets(due) = [];
        safetyPackets(dueSafety) = [];
    end

    occupiedHistory(historyCursor,:) = occupancy(slot,:);
    historyCursor = mod(historyCursor,size(occupiedHistory,1)) + 1;
end

% Close all unfinished task outcomes exactly once.
for tid = 1:taskCount
    if ~ismember(task(tid).status,["completed","rejected","rejected_cpu_capacity", ...
            "rejected_helper_capacity","late_cpu","late_result","radio_expired", ...
            "expired_cpu_queue","preempted_by_safety"])
        task(tid).status = "deadline_expired";
    end
end

result.safety = array2table(safetyRows,'VariableNames', ...
    {'flow_id','seq','tx_id','rx_id','created_ms','deadline_ms','decoded_ms', ...
     'success_on_time','bytes','range_m','distance_bin'});
result.safety.success_on_time = logical(result.safety.success_on_time);
result.decisions = cell2table(decisionRows,'VariableNames', ...
    {'scenario_id','decision_id','time_ms','vehicle_id','task_id','action_mask', ...
     'chosen_action','risk_upper','policy_id','decision_runtime_ms'});
result.radio = cell2table(radioRows,'VariableNames', ...
    {'scenario_id','time_ms','ue_id','packet_id','traffic_class','priority', ...
     'subchannel_id','reservation_counter','stage','sinr_db','tx_attempt','decoded','receiver_id'});
result.tasks = buildTaskTable(trace,task);
result.summary = struct('seed',trace.seed,'density',trace.density,'policy',policy, ...
    'safety_opportunities',height(result.safety), ...
    'safety_success',sum(result.safety.success_on_time), ...
    'tasks',height(result.tasks),'tasks_completed',sum(result.tasks.completed_on_time), ...
    'radio_packets',numel(unique(result.radio.packet_id)), ...
    'fragments',sum(result.tasks.input_fragments+result.tasks.output_fragments), ...
    'cbr',mean(occupancy,'all'));
end

function [action,helper,risk] = chooseAction(policy,tid,origin,candidates,cpuQueue,cpuActive,trace,cfg,cbr,x,y)
helper=0; risk=min(1, max(0, 0.002 + 0.015*cbr + 0.003*(trace.tasks.input_bytes(tid)/cfg.radio.max_transport_block_bytes)));
localFinish=(numel(cpuQueue{origin})+double(cpuActive(origin)>0)+1)*trace.tasks.cycles(tid)/trace.cpu_rate(origin)*1000;
localFeasible=trace.tasks.release_ms(tid)+localFinish <= trace.tasks.deadline_ms(tid);
if policy == "no_offload"
    if localFeasible, action="local"; else, action="reject"; end; return
end
if isempty(candidates)
    if localFeasible, action="local"; else, action="reject"; end; return
end
loads=arrayfun(@(v) numel(cpuQueue{v})+double(cpuActive(v)>0),candidates);
dist=arrayfun(@(v) hypot(x(v)-x(origin),y(v)-y(origin)),candidates);
if policy == "edf", score=loads*1000+dist; else, score=loads*100+dist; end
[~,j]=min(score); helper=candidates(j);
switch policy
    case "cbr_threshold"
        if cbr <= cfg.control.cbr_threshold, action="offload"; elseif localFeasible, action="local"; else, action="reject"; end
    case "risk_heuristic"
        if cbr >= cfg.control.risk_ood_cbr
            action="reject";
        elseif risk+cfg.control.risk_margin <= cfg.control.risk_harm_limit
            action="offload";
        elseif localFeasible
            action="local";
        else
            action="reject";
        end
    otherwise
        action="offload";
end
end

function candidates=helperCandidates(origin,x,y,maxRange,maxHelpers)
d=hypot(x-x(origin),y-y(origin)); d(origin)=inf;
[sorted,idx]=sort(d); idx=idx(sorted<=maxRange); candidates=idx(1:min(maxHelpers,numel(idx)))';
end

function [out,nextId]=makeComputePackets(nextId,tid,src,dst,bytes,stage,created,deadline,tb,currentSlot,selectionSlots,nSub,seed,policy)
n=ceil(bytes/tb); template=struct('id',0,'traffic_class',"compute",'src',src,'dst',dst, ...
    'created_ms',created,'deadline_ms',deadline,'bytes',0,'task_id',tid,'stage',stage, ...
    'fragment',0,'fragments',n,'flow_id',1000000+tid,'seq',0,'rx_ids',dst, ...
    'scheduled_slot',0,'subchannel',0,'priority',2);
out=repmat(template,n,1);
for f=1:n
    nextId=nextId+1; out(f).id=nextId; out(f).fragment=f; out(f).seq=f;
    out(f).bytes=min(tb,bytes-(f-1)*tb);
    baseOffset=floor(stableUniform(seed,"compute-slot",tid,stage,f)*selectionSlots);
    if policy=="priority_only", baseOffset=min(selectionSlots-1,baseOffset+1); end
    out(f).scheduled_slot=currentSlot+1+baseOffset;
    out(f).subchannel=1+floor(stableUniform(seed,"compute-subch",tid,stage,f)*nSub);
end
end

function T=buildTaskTable(trace,state)
n=height(trace.tasks); action=strings(n,1); destination=zeros(n,1); status=strings(n,1);
inputDelivered=nan(n,1); cpuStart=nan(n,1); cpuFinish=nan(n,1); resultDelivered=nan(n,1);
inputFragments=zeros(n,1); outputFragments=zeros(n,1); decisionId=zeros(n,1);
for i=1:n
    action(i)=state(i).action; destination(i)=state(i).helper; status(i)=state(i).status;
    inputDelivered(i)=state(i).input_delivered_ms; cpuStart(i)=state(i).cpu_start_ms;
    cpuFinish(i)=state(i).cpu_finish_ms; resultDelivered(i)=state(i).result_delivered_ms;
    inputFragments(i)=state(i).input_fragments; outputFragments(i)=state(i).output_fragments;
    decisionId(i)=state(i).decision_id;
end
T=table(trace.tasks.task_id,trace.tasks.origin_id,trace.tasks.release_ms,trace.tasks.deadline_ms, ...
    trace.tasks.input_bytes,trace.tasks.output_bytes,trace.tasks.cycles,trace.tasks.value, ...
    action,destination,decisionId,inputDelivered,cpuStart,cpuFinish,resultDelivered, ...
    status,status=="completed",inputFragments,outputFragments, ...
    'VariableNames',{'task_id','origin_id','release_ms','deadline_ms','in_bytes','out_bytes', ...
    'cycles','value','action','destination','decision_id','input_delivered_ms','cpu_start_ms', ...
    'cpu_finish_ms','result_delivered_ms','failure_reason','completed_on_time','input_fragments','output_fragments'});
end

function d=wrappedDistance(a,b,L)
d=abs(a-b); d=min(d,L-d);
end

function pl=pathLossDb(distance_m,freq_ghz)
pl=32.4+20*log10(freq_ghz)+20*log10(distance_m);
end

function mw=db2powLocal(dbm)
mw=10.^(dbm/10);
end

