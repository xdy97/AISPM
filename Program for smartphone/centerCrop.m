function videoResized = centerCrop(video,inputSize)
%   Copyright 2020-2021 The MathWorks, Inc.

sz = size(video);
videoTmp = video;

if sz(1) < sz(2)
    % Video is landscape
    idx = floor((sz(2) - sz(1))/2);
    videoTmp(:,1:(idx-1),:,:) = [];
    videoTmp(:,(sz(1)+1):end,:,:) = [];
    
elseif sz(2) < sz(1)
    % Video is portrait
    idx = floor((sz(1) - sz(2))/2);
    videoTmp(1:(idx-1),:,:,:) = [];
    videoTmp((sz(2)+1):end,:,:,:) = [];
end

videoResized = imresize(videoTmp,inputSize(1:2));
try
    videoResized = reshape(videoResized, inputSize(1), inputSize(2), inputSize(3), []);
end
end