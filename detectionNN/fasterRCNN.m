function fasterRCNN(input_opts, model_opts, output_opts, varargin)
%% Main body of the function

% Header options for the detector
    opts.gpu = [] ;
    if(model_opts.gpu_switch == 'gpu')
        opts.gpu = [1];
    end
    opts.modelPath = '';
    opts.roiVar = 'proposal'; % for older models: 'proposal'
    opts.scale = 720;
    opts.nmsThresh = 0.35;
    opts.confThresh = 0.80;
    opts.maxScale = 1000;
    
    % the HDR model is autonn and the LDR model is dagnn
    if (strcmp(input_opts.format, 'hdr') == 1)
        opts.wrapper = 'autonn';  
    else
        opts.wrapper = 'dagnn';  
    end
    
    opts = vl_argparse(opts, varargin);

% Pascal VOC 2007 challenge classes for common objects in context
    classes = {'background', 'aeroplane', 'bicycle', 'bird', ...........................
               'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', ...
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', ..........
               'sofa', 'train', 'tvmonitor'};

% Model selection (in case there is no path in the main framework file:

    if (strcmp(input_opts.format, 'hdr') == 1)        
        modelName = 'fasterRCNN_hdr.mat';
    else
        modelName = 'fasterRCNN_ldr.mat';
    end
    
    paths = {opts.modelPath,modelName,fullfile(vl_rootnn, 'data', 'models', modelName)} ;
    ok = find(cellfun(@(x) exist(x, 'file'), paths), 1) ;

    if isempty(ok)
        error('\n The model path is empty. Please specify the correct path');
    else
        opts.modelPath = paths{ok} ;
    end 
    
% Load the network with the chosen wrapper and move it to GPU
    fprintf('\n Loading model... \n Please wait as this is a one time operation');
    net = loadPretrainedModel(opts) ;
    
% Switching between image and video modes
   if(strcmp(input_opts.type, 'image') == 1)
        evaluate_single_image(input_opts,classes, net, opts, output_opts); 
    elseif(strcmp(input_opts.type, 'sequence') == 1)
        evaluate_frame_sequence(input_opts, classes, net, opts, output_opts);
    elseif(strcmp(input_opts.type, 'video') == 1)
        evaluate_video_sequence(input_opts, classes, net, opts, output_opts);
   end 
    
end   

function evaluate_single_image(input_opts, classes, net, opts, output_opts)
%% Detect objects in single image and display output + write output
    % move network to GPU is switched on..
    if numel(opts.gpu) > 0
       gpuDev = gpuDevice(opts.gpu) ; 
       net.move('gpu') ; 
    end       
    %% object detection for HDR input image
    if (strcmp(input_opts.format, 'hdr') == 1)
        hdr = single(exrread(input_opts.path));                                                               
        img_detected = detectionOperation(hdr, classes, net, opts, gpuDev);
    end 
    %% writing the file and showing the output
    if (strcmp(input_opts.format, 'hdr') == 1)
        exrwrite(img_detected, output_opts.path);
    elseif(strcmp(input_opts.format, 'ldr') == 1 || strcmp(input_opts.format, 'tmo') == 1)
        imwrite(img_detected, output_opts.path);
    end
    imshow(img_detected);
    
    %% moving the net from GPU to CPU
    if numel(opts.gpu) > 0
        net.move('cpu');
    end
end

function evaluate_frame_sequence(input_opts, classes, net, opts, output_opts)
    if ~exist(output_opts.path, 'dir')
        mkdir(output_opts.path);
        fprintf('\n Output directory did not exist. Created output folder');
    end
    
    if numel(opts.gpu) > 0
       gpuDev = gpuDevice(opts.gpu) ; 
       net.move('gpu') ; 
    end  
     %% Native HDR based object detection
    if (strcmp(input_opts.format, 'hdr') == 1)                        
        filelist = dir(fullfile(input_opts.path, '*.exr'));        
        tic; % start the clock to count total execution time
        for i = 1 : numel(filelist)            
            hdr = single(exrread(fullfile(filelist(i).folder, filelist(i).name)));                                                                                                                                  
            hdr = ClampImg(hdr, 1e-3, 255);
            img_detected = detectionOperation(hdr, classes, net, opts, gpuDev);         
            % write output to output folder - filename_postfix "detected"
            output_filename = fullfile(output_opts.path, filelist(i).name);            
            exrwrite(img_detected, output_filename);
            fprintf('\n Frame %s completed..', filelist(i).name);
        end  
        toc; % end clock to see elapsed time
    end
    
    %% TMO based object detection
    if (strcmp(input_opts.format, 'tmo') == 1)                        
        filelist = dir(fullfile(input_opts.path, '*.exr'));        
        tic; % start the clock to count total execution time
        for i = 1 : numel(filelist)            
            hdr = single(exrread(fullfile(filelist(i).folder, filelist(i).name)));
            hdr = ClampImg(hdr, 1e-4, 255);
            preprocessed_tmo =  single(real(lin2srgb(ReinhardTMO(hdr)))*255.0);
            img_detected = detectionOperation(preprocessed_tmo, classes, net, opts, gpuDev);         
            % write output to output folder - filename_postfix "detected"
            output_filename = fullfile(output_opts.path, sprintf('%05d.jpg', (i-1)));            
            imwrite(img_detected, output_filename, 'quality', 100);
            fprintf('\n Frame %s completed..', filelist(i).name);
        end  
        toc; % end clock to see elapsed time
    end
    
    %% LDR based object detection
    if (strcmp(input_opts.format, 'ldr') == 1)
        filelist = dir(fullfile(input_opts.path, 'ldr_*.jpg'));        
        tic; % start the clock to count total execution time
        for i = 1 : numel(filelist)            
            ldr = single(imread(fullfile(filelist(i).folder, filelist(i).name))); 
            img_detected = detectionOperation(ldr, classes, net, opts, gpuDev);
            output_filename = fullfile(output_opts.path, sprintf('%05d.jpg', (i-1)));            
            imwrite(img_detected, output_filename, 'quality', 100);
            fprintf('\n Frame %s completed..', filelist(i).name);
        end
        toc; % start the clock to count total execution time
    end
    
    %% Move net from GPU to CPU
    if numel(opts.gpu) > 0
        net.move('cpu');
    end    
    fprintf('\n Image Sequence detection complete..\n');    
end

function evaluate_video_sequence(input_opts, classes, net, opts, output_opts)
%% function to detect objects in a video sequence (this is typically LDR)

    if ~exist(output_opts.path, 'dir')
        mkdir(output_opts.path);
        fprintf('\n Output directory did not exist. Created output folder');
    end
    
    if numel(opts.gpu) > 0
       gpuDev = gpuDevice(opts.gpu) ; 
       net.move('gpu') ; 
    end 
        
%% initializing the video file reader and writer (computer vision toolbox)
    videoFReader = vision.VideoFileReader(input_opts.path); 
    
    if (strcmp(output_opts.type, 'video') == 1)
        vid_fwrite = vision.VideoFileWriter(output_opts.path);
        vid_fwrite.FileFormat = 'MPEG4';    
        vid_fwrite.FileColorSpace = 'RGB';
        vid_fwrite.FrameRate = 25;
        vid_fwrite.Quality = 100;
    end 
%% read the video frames and perform detection on that
    frame_index = 0;
    tic;
    while ~isDone(videoFReader)        
        ldr_frame = single(im2uint8(videoFReader()));
        img_detected = detectionOperation(ldr_frame, classes, net, opts, gpuDev);
        if (strcmp(output_opts.type, 'video') == 1)
            step(vid_fwrite, im2uint8(im));
        elseif(strcmp(output_opts.type, 'sequence') == 1)
            output_filename = fullfile(output_opts.path, sprintf('%05d.jpg', (frame_index)));
            imwrite(im2uint8(img_detected), output_filename, 'quality', 100);                        
        end
        fprintf('\n Frame %05d written..', frame_index);
        frame_index = frame_index + 1;    
    end 
    
    toc;
    if (strcmp(output_opts.type, 'video') == 1)
        release(videoFReader); release(vid_fwrite);
    end
    
%% move the net back to CPU
    if numel(opts.gpu) > 0
        net.move('cpu');
    end    
    fprintf('\n Video file detection complete.. \n');
end 


function [img_detected] = detectionOperation(im,classes, net, opts, gpuDev)
    if strcmp(opts.wrapper, 'dagnn')
        clsIdx = net.getVarIndex('cls_prob') ;
        bboxIdx = net.getVarIndex('bbox_pred') ;
        roisIdx = net.getVarIndex(opts.roiVar) ;
        [net.vars([clsIdx bboxIdx roisIdx]).precious] = deal(true) ;
    end

    % resize to meet the faster-rcnn size criteria
    imsz = [size(im,1) size(im,2)] ; maxSc = opts.maxScale ; 
    factor = max(opts.scale ./ imsz) ; 
    if any((imsz * factor) > maxSc), factor = min(maxSc ./ imsz) ; end
    newSz = factor .* imsz ; imInfo = [ round(newSz) factor ] ;

    % resize and subtract mean  
    data = imresize(im, factor, 'bicubic') ;  
    data = bsxfun(@minus, data, net.meta.normalization.averageImage) ;

    if numel(opts.gpu) > 0         
        data = gpuArray(data) ;
    end

    % set inputs
    imInfo = [ round(newSz) 1.00 ] ;
    sample = {'data', data, 'im_info', imInfo} ;
    net.meta.classes.name = classes ;


    % run network and retrieve results
    switch opts.wrapper
    case 'dagnn'       
      net.eval(sample) ;      
      probs = gather(squeeze(net.vars(clsIdx).value)) ;
      deltas = gather(squeeze(net.vars(bboxIdx).value)) ;
      boxes = gather(net.vars(roisIdx).value(2:end,:)' / imInfo(3));
    case 'autonn'
      net.eval(sample, 'test') ;
      probs = gather(squeeze(net.getValue('cls_prob'))) ;
      deltas = gather(squeeze(net.getValue('bbox_pred'))) ;
      boxes = gather(net.getValue('proposal')) ;
      boxes = boxes(2:end,:)' / imInfo(3) ;
    end    

    % Free up the GPU allocation
    if numel(opts.gpu) > 0  
      data = gather(data);  
      wait(gpuDev);      
    end

    im = im / 255 ; % normalize the image
    %% Visualize results for one class at a time
    box_index = 1;
    for i = 2:numel(classes)
        c = find(strcmp(classes{i}, net.meta.classes.name)) ;
        cprobs = probs(c,:) ;
        cdeltas = deltas(4*(c-1)+(1:4),:)' ;

        cboxes = bbox_transform_inv(boxes, cdeltas);
        cls_dets = [cboxes cprobs'] ;

        keep = bbox_nms(cls_dets, opts.nmsThresh) ;
        cls_dets = cls_dets(keep, :) ;

        sel_boxes = find(cls_dets(:,end) >= opts.confThresh) ;
        
        if numel(sel_boxes) == 0
            continue ; 
        end

        %% custom code to shortlist categories and store coordinates        
        if numel(sel_boxes >=1)   
            for sel_index = 1 : numel(sel_boxes)
                selected_boxes(box_index, 1:5) = num2cell(cls_dets(sel_boxes(sel_index), :));
                selected_boxes(box_index, 6) = cellstr(classes{i});
                box_index = box_index + 1;
            end
        end  
        selected_boxes(all(cellfun('isempty',selected_boxes),2),:) = [];
        CM = spring(size(selected_boxes, 1));
    end       

    %% draw bounding boxes for the objects on the image
    %figure(1) ; 
    if (box_index > 1)
        for ii = 1 : size(selected_boxes, 1)
            bounding_box = selected_boxes(ii, 1:end);
            tb.x = cell2mat(bounding_box(1)) * 1/factor;
            tb.y = cell2mat(bounding_box(2)) * 1/factor;
            tb.width = cell2mat(bounding_box(3)) * 1/factor - tb.x;
            tb.height = cell2mat(bounding_box(4)) * 1/factor - tb.y;
            tb.prob = cell2mat(bounding_box(5)); tb.str = bounding_box{6}; 
            rectangle = [tb.x tb.y tb.width tb.height];        
            im = insertShape(im, 'Rectangle', rectangle, 'LineWidth', 4, 'Color', CM(ii,:));            
            text_str = sprintf('%s: %.2f', tb.str, tb.prob);
            im = insertText(im, [double(tb.x) double(tb.y-25)], text_str, 'FontSize', 14, 'TextColor', 'black', 'BoxColor', CM(ii,:));       
        end 
    end

    img_detected = im;
end