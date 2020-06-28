function [detectARR] = detectionOutputResults()
%% Create table to add detection based AP
        detect_table = table(cell(numel(filelist), 3), 'VariableNames', {'frame', 'car', 'person'}); 

%% pass the labels and bounding boxes to the function
    cellARR = tableManipulation(frameNo, label, rectangle); 
    celltbl = cell2table(cellARR);
    detect_table = vertcat(detect_table, celltbl);
    
%% table manipulation
end

function [cellArr] = tableManipulation(frameNo, label, rectangle)
    cls = unique(label);
    cellArr = cell([1, (1+ numel(cls))]);
    cellArr{1} = frameNo;
    
    for i = 1 : numel(cls)
        idx = strfind(label, cls{i});
        idx = find(not(cellfun('isempty', idx)));
        
        for j = 1 : numel(idx)
            val = idx(j);
            cellArr{:,(i+1)} = vertcat(cellArr{:,(i+1)}, rectangle(val, :));
        end
    end
end