function stationManagement = spv2xWiLabReception(stationManagement,timeManagement,txIDs,fate,~)
%SPV2XWILABRECEPTION Consume WiLab PHY receiver fate rows for tagged packets.
s=stationManagement.spv2x; nowMs=timeManagement.timeNow*1000;
for tx=txIDs(:)'
    meta=s.current(tx); if ~meta.valid, continue; end
    rows=fate(fate(:,1)==tx,:);
    for i=1:size(rows,1)
        rx=rows(i,2); success=logical(rows(i,5)) && nowMs<=meta.deadline_ms;
        s.radio_rows(end+1,:)={s.scenario_id,nowMs,tx,meta.packet_id,char(meta.kind), ...
            meta.task_id,char(meta.stage),meta.fragment,rx,rows(i,4),rows(i,3),logical(rows(i,5)),char(s.world)};
        if meta.kind=="safety"
            ix=find(s.safety_rows(:,12)==meta.packet_id & s.safety_rows(:,4)==rx,1);
            if ~isempty(ix) && success, s.safety_rows(ix,7)=nowMs; s.safety_rows(ix,8)=1; end
        elseif rx==meta.dst && success
            tid=meta.task_id;
            if meta.stage=="input"
                s.tasks(tid).input_received=s.tasks(tid).input_received+1;
                if s.tasks(tid).input_received==s.tasks(tid).input_fragments
                    s.tasks(tid).input_delivered_ms=nowMs; s.tasks(tid).status="queued_remote";
                    s.cpu_queue{meta.dst}(end+1)=tid;
                end
            else
                s.tasks(tid).output_received=s.tasks(tid).output_received+1;
                if s.tasks(tid).output_received==s.tasks(tid).output_fragments
                    s.tasks(tid).result_delivered_ms=nowMs;
                    if nowMs<=s.tasks(tid).deadline_ms, s.tasks(tid).status="completed"; else, s.tasks(tid).status="late_result"; end
                end
            end
        end
    end
    if stationManagement.pckTxOccurring(tx)>=stationManagement.cv2xNumberOfReplicas(tx)
        s.current(tx)=s.empty_meta;
    end
end
stationManagement.spv2x=s;
end

