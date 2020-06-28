function [gt_tbl] = convertGTLabels(gTruth)
%% Script to convert GroundTruth to a table for comparison
frameNo = string(ceil(gTruth.LabelData.Time(:)./(33.34/1000)));
frames = zeros([numel(frameNo), 1]);
for i = 1 : numel(frameNo)
    [f, ~] = strsplit(frameNo(i));
    frames(i, 1) = str2double(f(1));
end

gt_tbl = cell2table(cell(numel(frameNo), 3), 'VariableNames', {'frames', 'person', 'car'});
gt_tbl.car = gTruth.LabelData.car;
gt_tbl.person = gTruth.LabelData.person;
gt_tbl.frames = frames;
end
