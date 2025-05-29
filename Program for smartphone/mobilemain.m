load('net.mat')
inputSize = [224,224];

filename = "p1_1.mp4";
video = readVideo(filename);

numFrames = size(video,4);
figure
for i = 1:numFrames
    frame = video(:,:,:,i);
    imagesc(frame(:,:,2));
    drawnow
end

video = centerCrop(video,inputSize);
YPred = classify(net,{video})
