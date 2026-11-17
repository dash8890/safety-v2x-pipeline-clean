function sensedPower_MHz = sensedPowerCV2X(stationManagement,sinrManagement,appParams,phyParams)
% This function calculates the power sensed by LTE nodes in a subframe
% The output does not include the interference from 11p nodes, if present

NbeaconsF = appParams.NbeaconsF;
sensedPower_MHz = zeros(NbeaconsF,length(stationManagement.activeIDsCV2X));

if ~isempty(stationManagement.transmittingIDsCV2X)

    activeIDsCV2X = stationManagement.activeIDsCV2X;

    % Calculate the beacon resources used in the time and frequency domains
    % Find not assigned BRid
    %idNOT = (stationManagement.BRid(:,1)<=0);
    % Calculate BRidT = vector of BRid in the time domain
    %BRidT = ceil(BRid/NbeaconsF);
    %BRidT(idNOT) = -1;
    % Calculate BRidF = vector of BRid in the frequency domain
    %BRidF = mod(stationManagement.BRid(:,1)-1,NbeaconsF)+1;
    %BRidF(idNOT) = -1;
    
    allGlobal=stationManagement.indexInActiveIDs_ofLTEnodes;
    txLocal=stationManagement.indexInActiveIDsOnlyLTE_OfTxLTE(:);
    txGlobal=allGlobal(txLocal); fused=stationManagement.transmittingFusedLTE(1:numel(txLocal));
    fused=fused(:);
    rxPsums=zeros(NbeaconsF,numel(activeIDsCV2X));
    for resource=1:NbeaconsF
        selected=find(fused==resource);
        if isempty(selected), continue; end
        rxPsums(resource,:)=sum(sinrManagement.P_RX_MHz(allGlobal,txGlobal(selected)),2)';
    end
    for iTx=1:numel(txLocal)
        resource=fused(iTx); local=txLocal(iTx); globalIndex=txGlobal(iTx);
        rxPsums(resource,local)=rxPsums(resource,local)-sinrManagement.P_RX_MHz(globalIndex,globalIndex) + ...
            phyParams.Ksi*phyParams.P_ERP_MHz_CV2X(activeIDsCV2X(local));
    end
    sensedPower_MHz=phyParams.IBEmatrixData*rxPsums;
end
