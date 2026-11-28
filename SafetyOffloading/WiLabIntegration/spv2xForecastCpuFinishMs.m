function finishMs=spv2xForecastCpuFinishMs(s,vehicle,nowS,newCycles,tentative)
%SPV2XFORECASTCPUFINISHMS Append-to-queue CPU finish time under residual rate.
% tentative may be empty or a struct with cpu_queue / cpu_active / cpu_finish_s overlays.
if nargin<5 || isempty(tentative)
    queue=s.cpu_queue{vehicle};
    active=s.cpu_active(vehicle);
    finishS=s.cpu_finish_s(vehicle);
else
    if isfield(tentative,'cpu_queue') && numel(tentative.cpu_queue)>=vehicle && ~isempty(tentative.cpu_queue{vehicle})
        queue=tentative.cpu_queue{vehicle};
    else
        queue=s.cpu_queue{vehicle};
    end
    if isfield(tentative,'cpu_active') && numel(tentative.cpu_active)>=vehicle
        active=tentative.cpu_active(vehicle);
    else
        active=s.cpu_active(vehicle);
    end
    if isfield(tentative,'cpu_finish_s') && numel(tentative.cpu_finish_s)>=vehicle
        finishS=tentative.cpu_finish_s(vehicle);
    else
        finishS=s.cpu_finish_s(vehicle);
    end
    if isfield(tentative,'extra_cycles') && numel(tentative.extra_cycles)>=vehicle
        newCycles=newCycles+tentative.extra_cycles(vehicle);
    end
end
backlogCycles=sum(arrayfun(@(id)s.tasks(id).cycles,queue));
if active>0
    backlogCycles=backlogCycles+max(0,finishS-nowS)*s.cpu_rate(vehicle);
end
finishMs=nowS*1000+(backlogCycles+newCycles)/s.cpu_rate(vehicle)*1000;
end
