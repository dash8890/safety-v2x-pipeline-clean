function stationManagement = spv2xWiLabReception(stationManagement,timeManagement,txIDs,fate,~)
%SPV2XWILABRECEPTION Consume WiLab PHY receiver fate rows for tagged packets.
s=stationManagement.spv2x;
if startsWith(string(s.policy),"workflow_")
    stationManagement=spv2xWorkflowBridge('reception',stationManagement,timeManagement,txIDs,fate);
    return
end
nowMs=timeManagement.timeNow*1000;
for tx=txIDs(:)'
    meta=s.current(tx); if ~meta.valid, continue; end
    rows=fate(fate(:,1)==tx,:);
    for i=1:size(rows,1)
        rx=rows(i,2); success=logical(rows(i,5)) && nowMs<=meta.deadline_ms;
        if s.log_radio_events && meta.kind=="compute"
            s.radio_rows(end+1,:)={s.scenario_id,nowMs,tx,meta.packet_id,char(meta.kind), ...
                meta.task_id,char(meta.stage),meta.fragment,rx,rows(i,4),rows(i,3),logical(rows(i,5)),char(s.world)};
        end
        if meta.kind=="safety"
            first=double(s.safety_row_start(meta.packet_id)); count=double(s.safety_row_count(meta.packet_id));
            if first>0
                local=find(s.safety_rows(first:first+count-1,4)==rx,1);
                if ~isempty(local) && success
                    ix=first+local-1; s.safety_rows(ix,7)=nowMs; s.safety_rows(ix,8)=1;
                end
            end
        elseif rx==meta.dst && success
            tid=meta.task_id;
            if meta.stage=="input"
                s.tasks(tid).input_received=s.tasks(tid).input_received+1;
                if s.tasks(tid).input_received==s.tasks(tid).input_fragments
                    s.tasks(tid).input_delivered_ms=nowMs; s.tasks(tid).status="queued_remote";
                    owner=meta.dst;
                    if s.cpu_active(owner)==0 && nowMs<s.tasks(tid).deadline_ms
                        s.cpu_active(owner)=tid; s.tasks(tid).cpu_start_ms=nowMs;
                        s.cpu_finish_s(owner)=timeManagement.timeNow+s.tasks(tid).cycles/s.cpu_rate(owner);
                        s.next_tick_s=min(s.next_tick_s,s.cpu_finish_s(owner));
                    else
                        s.cpu_queue{owner}(end+1)=tid;
                    end
                end
            else
                s.tasks(tid).output_received=s.tasks(tid).output_received+1;
                if s.tasks(tid).output_received==s.tasks(tid).output_fragments
                    s.tasks(tid).result_delivered_ms=nowMs;
                    if nowMs<=s.tasks(tid).deadline_ms
                        s.tasks(tid).status="completed";
                        if ~isfield(s,'reward_rows'), s.reward_rows=cell(0,5); end
                        s.reward_rows(end+1,:)={s.scenario_id,tid,nowMs,1,"completion"}; %#ok<AGROW>
                    else
                        s.tasks(tid).status="late_result";
                    end
                end
            end
        end
    end
    if stationManagement.pckTxOccurring(tx)>=stationManagement.cv2xNumberOfReplicas(tx)
        s.current(tx)=s.empty_meta;
        if ~isempty(s.meta_queue{tx})
            s.current(tx)=s.meta_queue{tx}(1); s.meta_queue{tx}(1)=[];
        end
    end
end
stationManagement.spv2x=s;
end

