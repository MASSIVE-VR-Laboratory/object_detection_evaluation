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
 %
 % Description: This is the main controlling script of the object detection
 % framework where given the main input parameters, object detection
 % scripts are executed to detect objects in 3 scenarios:
 %  1) Native HDR images / video frames
 %  2) Tone Mapped LDR images / frames
 %  3) LDR images / video frames. 
 % Author: Ratnajit Mukherjee, ratnajit.mukherjee@inesctec.pt
 % Date: December, 2018
%%%
%% Header options
input_opts.type = 'sequence'; % there are 3 values: image, sequence, video
input_opts.format = 'tmo'; % there are 3 values: ldr, tmo, hdr
input_opts.path = 'D:\Finalised_Sequences\GarageDoor\';

model_opts.type = 'fasterRCNN'; % there are 2 options: ssd and fasterRCNN
model_opts.path = '';
model_opts.fit_size = 512; % there are 2 options: 300/512 for SSD.
model_opts.gpu_switch = 'gpu'; % there are 2 options: gpu/cpu

output_opts.type='sequence'; % there are 3 values: image, sequence, video
output_opts.path= 'D:\detection_output_fasterRCNN\GarageDoor\tmo_detection\';

%% controlling portion of the script
if (strcmp(model_opts.type, 'ssd') == 1)
    singleShotDetector(input_opts, model_opts, output_opts); 
elseif (strcmp(model_opts.type, 'fasterRCNN') == 1)
    fasterRCNN(input_opts, model_opts, output_opts);
else
    error('\n ** WRONG DETECTION ALGORITHM SELECTED. PLEASE TRY AGAIN \n');
end

fprintf('\n *******FINISHED DETECTION TASK*******\n');