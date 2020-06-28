%%%
 % License:
 % -----------------------------------------------------------------------------
 % Copyright (c) 2018, INESCTEC, Portugal, ONR London.
 % All rights reserved.
 % 
 % Redistribution and use in source and binary forms, with or without 
 % modification, are permitted provided that the following conditions are met:
 % 
 % 1. Redistributions of source code must retain the above copyright notice, 
 %    this list of conditions and the following disclaimer.
 % 
 % 2. Redistributions in binary form must reproduce the above copyright notice,
 %    this list of conditions and the following disclaimer in the documentation
 %    and/or other materials provided with the distribution.
 % 
 % 3. Neither the name of the copyright holder nor the names of its contributors
 %    may be used to endorse or promote products derived from this software 
 %    without specific prior written permission.
 % 
 % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS %AS IS% 
 % AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 % IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 % ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 % LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 % CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 % SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 % INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 % CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 % ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 % POSSIBILITY OF SUCH DAMAGE.
 % -----------------------------------------------------------------------------
 % Description: Controlling function to use MatConvNet SSD for single
 % images / video frame sequences / video frames. 
 %
 % Author: Ratnajit Mukherjee, ratnajit.mukherjee@inesctec.pt
 % Date: December, 2018
%%%

function singleShotDetector(input_opts, model_opts, output_opts, varargin)
%% controlling function for Single Shot Multibox detector
% Header options for the detector
    opts.gpu = [];
    if(model_opts.gpu_switch == 'gpu')
        opts.gpu = [1];
    end
    opts.modelPath = model_opts.path;
    opts.wrapper = 'autonn' ;
    opts = vl_argparse(opts, varargin) ;

% Pascal VOC 2007 challenge classes 
    classes = { 'background', 'aeroplane', 'bicycle', 'bird', ...
                'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',...
                'diningtable', 'dog', 'horse', 'motorbike', 'person', ...
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor' };

% Model selection (in case there is no path in the main framework file
    if (strcmp(input_opts.format, 'hdr') == 1)
        if(model_opts.fit_size == 512)
            modelName = 'ssd-mcn-pascal-vggvd-512-hdr.mat';
        elseif(model_opts.fit_size == 300)
            modelName = 'ssd-mcn-pascal-vggvd-300-hdr.mat';
        end
    else
        if(model_opts.fit_size == 512)
            modelName = 'ssd-mcn-pascal-vggvd-512.mat';
        elseif(model_opts.fit_size == 300)
            modelName = 'ssd-mcn-pascal-vggvd-300.mat';
        end
    end
        
    paths = {opts.modelPath, ...
           modelName, ...
           fullfile(vl_rootnn, 'data/models', modelName), ...
           fullfile(vl_rootnn, 'data', 'models-import', modelName)} ;
    ok = find(cellfun(@(x) exist(x, 'file'), paths), 1);
    
    if isempty(ok)
       error('\n The model path is empty. Please specify the corrent path');
    else
        opts.modelPath = paths{ok};
    end
         
% Load the network with the chosen wrapper
    fprintf('\n Loading model... \n Please wait as this is a one time operation');
    net = loadPretrainedModel(opts);    
    
% Switching between image and video modes
    if(strcmp(input_opts.type, 'image') == 1)
        evaluate_single_image(input_opts, model_opts, classes, net, opts, output_opts); 
    elseif(strcmp(input_opts.type, 'sequence') == 1)
        evaluate_frame_sequence(input_opts, model_opts, classes, net, opts, output_opts);
    elseif(strcmp(input_opts.type, 'video') == 1)
        evaluate_video_sequence(input_opts, model_opts, classes, net, opts, output_opts);
    end    
end

function evaluate_single_image(input_opts, model_opts, classes, net, opts, output_opts)
%% Detect objects in single image and display output + write output
    % move network to GPU is switched on..
    if numel(opts.gpu) > 0
       gpuDev = gpuDevice(opts.gpu) ; 
       net.move('gpu') ; 
    end       
    %% object detection for HDR input image
    if (strcmp(input_opts.format, 'hdr') == 1)
        hdr = single(exrread(input_opts.path));                                                
        resized_hdr = imresize(hdr, [model_opts.fit_size model_opts.fit_size], 'bicubic');
        resized_hdr = ClampImg(resized_hdr, 1e-3, max(resized_hdr(:)));
        preprocessed_hdr = single(hdrPreprocessing(resized_hdr));        
        img_detected= detectionOperation(preprocessed_hdr, hdr, classes, net, opts, gpuDev);
    end  
    
    %% object detection for TMO with input HDR
    if (strcmp(input_opts.format, 'tmo') == 1)
        hdr = single(exrread(input_opts.path));                    
        preprocessed_tmo =  single(real(lin2srgb(ReinhardTMO(hdr)))*255.0);
        resized_tmo = imresize(preprocessed_tmo, [model_opts.fit_size model_opts.fit_size], 'bicubic');
        resized_tmo = ClampImg(resized_tmo, 1e-3, max(resized_tmo(:)));                  
        img_detected = detectionOperation(resized_tmo, preprocessed_tmo, classes, net, opts, gpuDev); 
    end
    
    %% object detection for LDR input
    if (strcmp(input_opts.format, 'ldr') == 1)
        ldr = single(imread(input_opts.path));                                 
        resized_ldr = imresize(ldr, [model_opts.fit_size model_opts.fit_size], 'bicubic');
        resized_ldr = ClampImg(resized_ldr, 1e-3, max(resized_ldr(:)));                  
        img_detected = detectionOperation(resized_ldr, ldr, classes, net, opts, gpuDev); 
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

function evaluate_frame_sequence(input_opts, model_opts, classes, net, opts, output_opts)
%% function to detect objects in an image sequence
    % check whether you have the output folder: if NO then CREATE
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
        % start the clock to count total execution time
        tic;
        for i = 1 : numel(filelist)            
            hdr = single(exrread(fullfile(filelist(i).folder, filelist(i).name)));                                                
            resized_hdr = imresize(hdr, [model_opts.fit_size model_opts.fit_size], 'bicubic');
            resized_hdr = ClampImg(resized_hdr, 1e-3, max(resized_hdr(:)));
            preprocessed_hdr = single(hdrPreprocessing(resized_hdr));        
            img_detected = detectionOperation(preprocessed_hdr, hdr, classes, net, opts, gpuDev);            
            % write output to output folder - filename_postfix "detected"
            output_filename = fullfile(output_opts.path, filelist(i).name);            
            exrwrite(img_detected, output_filename);
            fprintf('\n Frame %s completed..', filelist(i).name);
        end  
        toc;
        % end clock to see elapsed time
    end
    
    %% TMO based object detection
    if (strcmp(input_opts.format, 'tmo') == 1)
        filelist = dir(fullfile(input_opts.path, '*.exr'));
        % start the clock to count total execution time
        tic;
        for i = 1 : numel(filelist)            
            hdr = real(single(exrread(fullfile(filelist(i).folder, filelist(i).name))));                    
            preprocessed_tmo =  single(real(lin2srgb(ReinhardTMO(hdr)))*255.0);
            resized_tmo = imresize(preprocessed_tmo, [model_opts.fit_size model_opts.fit_size], 'bicubic');
            resized_tmo = ClampImg(resized_tmo, 1e-3, max(resized_tmo(:)));                  
            img_detected = detectionOperation(resized_tmo, preprocessed_tmo, classes, net, opts, gpuDev);            
            % write output to output folder - filename_postfix "detected"
            output_filename = fullfile(output_opts.path, sprintf('%05d.jpg', (i-1)));            
            imwrite(img_detected, output_filename, 'quality', 100);
            fprintf('\n Frame %s completed..', filelist(i).name);
        end
        toc;
        % start the clock to count total execution time
    end
    
    %% singleExposure/ldr object detection
    if (strcmp(input_opts.format, 'ldr') == 1)
        filelist = dir(fullfile(input_opts.path, 'ldr_*.jpg'));
        % start the clock to count total execution time
        tic;
        for i = 1 : numel(filelist)            
            ldr = single(imread(fullfile(filelist(i).folder, filelist(i).name)));                                 
            resized_ldr = imresize(ldr, [model_opts.fit_size model_opts.fit_size], 'bicubic');
            resized_ldr = ClampImg(resized_ldr, 1e-3, max(resized_ldr(:)));                  
            img_detected = detectionOperation(resized_ldr, ldr, classes, net, opts, gpuDev);            
            % write output to output folder - filename_postfix "detected"
            output_filename = fullfile(output_opts.path, sprintf('%05d.jpg', (i-1)));            
            imwrite(img_detected, output_filename, 'quality', 100);
            fprintf('\n Frame %s completed..', filelist(i).name);
        end
        toc;
        % start the clock to count total execution time
    end
    
    %% move the network from the GPU to CPU and release GPU resources
    if numel(opts.gpu) > 0
        net.move('cpu');
    end
    
    fprintf('\n Image Sequence detection complete..\n');    
end 

function evaluate_video_sequence(input_opts, model_opts, classes, net, opts, output_opts)
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
        resized_ldr = imresize(ldr_frame, [model_opts.fit_size model_opts.fit_size], 'bicubic');
        resized_ldr = ClampImg(resized_ldr, 1e-3, max(resized_ldr(:)));                  
        img_detected = detectionOperation(resized_ldr, ldr_frame, classes, net, opts, gpuDev);                    
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
    
    if numel(opts.gpu) > 0
        net.move('cpu');
    end
    
    fprintf('\n Video file detection complete.. \n');

end

function [img_detected] = detectionOperation(img, res_img, classes, net, opts, gpuDev)
%% Main detection operation (common to image/sequence/video frames)
    % moving the image to GPU
    if numel(opts.gpu) > 0
            img = gpuArray(img);            
    end
    
    % this assumes that the wrapper is autonn
    switch opts.wrapper
        case 'dagnn' 
            net.eval({'data', img}) ;
            preds = net.vars(end).value ;
        case 'autonn'
            net.eval({'data', img}, 'test') ;
            preds = net.getValue('detection_out') ;
    end

    [~, sortedIdx ] = sort(preds(:, 2), 'descend') ;
    preds = preds(sortedIdx, :);
    preds(preds(:,2)< 0.30, :) = [];
    % keep all predictions > = 45% confidence
    numKeep = size(preds, 1);

    % Extract the most confident predictions
    box = double(preds(1:numKeep,3:end)) ;
    confidence = preds(1:numKeep,2) ;
    label = classes(preds(1:numKeep,1)) ;

    if numel(opts.gpu) > 0
        img = gather(img);
        wait(gpuDev);
    end                    

    % insert bounding box and text into the image
    im = res_img / 255 ; CM = spring(numKeep); 
    x = box(:,1) * size(im, 2) ; y = box(:,2) * size(im, 1) ;
    width = box(:,3) * size(im, 2) - x ; height = box(:,4) * size(im, 1) - y ;
    rectangle = [x y width height];     
    
    im = insertShape(im, 'Rectangle', rectangle, 'LineWidth', 2, ...
                    'Color', CM(1:numKeep,:)) ;

    for ii = 1:numKeep
        str = sprintf('%s: %.2f', label{ii}, confidence(ii)) ;
        im = insertText(im, [x(ii), y(ii)- 25], str, 'FontSize', 10, ...                
                        'TextColor', 'black', 'BoxColor', CM(ii,:));
    end 
    
    img_detected = im;
end