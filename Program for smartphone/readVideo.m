function video = readVideo(filename, frameSize)

if coder.target('MATLAB')
    vr = VideoReader(filename);
else
    hwobj = jetson();
    vr = VideoReader(hwobj, filename, 'Width', frameSize(1), 'Height', frameSize(2));
end
% H = vr.Height;
H = 360;
% W = vr.Width;
W = 640;
C = 3;

% Preallocate video array
% numFrames = floor(vr.Duration * vr.FrameRate);
numFrames = 400;
video = zeros(H,W,C,numFrames);

% Read frames
i = 0;
while hasFrame(vr)
    i = i + 1;
    sz = [360 640];
    video(:,:,:,i) = imresize(readFrame(vr),'OutputSize',sz);
end

% Remove unallocated frames
if size(video,4) > i
    video(:,:,:,i+1:end) = [];
end

end