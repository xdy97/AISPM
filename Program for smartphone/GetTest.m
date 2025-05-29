
Data = VideoReader("D3_2.mp4");

A = read(Data,100);

g = [242.000000000000	499.000000000000
1293	602.000000000000]'

imshow(A(g(1,2):g(2,2),g(1,1):g(2,1)))

gg = 1;

numFrames = Data.NumFrames;


    p = VideoWriter(['test.mp4'],'MPEG-4');
    
    open(p)
    for i = 1:15:2000
        
        A = read(Data,i+1-1);
        
        frame = A(g(1,2):g(2,2),g(1,1):g(2,1),:);
        
        imagesc(frame(:,:,1));
        drawnow

        writeVideo(p,frame)
    end

    close(p)

