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
 % Description: Preprocessing the HDR frames with Dolby PQ curve. 
 %
 % Author: Ratnajit Mukherjee, ratnajit.mukherjee@inesctec.pt
 % Date: December, 2018
%%%
function [processed_hdr] = hdrPreprocessing(hdr)
%% function to apply Dolby PQ compression curve - for improved detection

    % scale input HDR to the desired range (this is the training range)        
    scaled_hdr = ((hdr - min(hdr(:)))./(max(hdr(:)) - min(hdr(:)))) * (255 - 1e-3) + 1e-3;
    
    % normalize the input HDR file
    normalized_hdr = zeros(size(scaled_hdr)); 
    for i=1:3
        mx(i) = max(max(scaled_hdr(:,:,i)));
        normalized_hdr(:,:,i) = scaled_hdr(:,:,i) / mx(i); 
    end
    
    % Non-linear function similar to PTF curve
    N = real(normalized_hdr.^(1/4));
    
    processed_hdr = zeros(size(N));   
    for i = 1 : 3
        N(:,:,i) = adapthisteq(N(:,:,i), 'NBins', 512, 'Range', 'original');
        processed_hdr(:,:,i) = N(:,:,i) .* (mx(i));
    end
end

