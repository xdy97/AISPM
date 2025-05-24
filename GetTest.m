
% 【构建数据集】对原始视频进行裁切选定

% 选取对象文件
Data = VideoReader("D3_2.mp4");

% 读取帧
A = read(Data,100);

% 获取角点（左上+右下），回车结束
g = [242.000000000000	499.000000000000
1293	602.000000000000]'

% 画选取区域图像
imshow(A(g(1,2):g(2,2),g(1,1):g(2,1)))


% 采样点
gg = 1;

% 视频帧数长度
numFrames = Data.NumFrames;



% 构建存放视频（文件名：
    p = VideoWriter(['test.mp4'],'MPEG-4');
    % 开放视频，准备存放视频帧2
    open(p)
    for i = 1:15:2000
        % 读取帧
        A = read(Data,i+1-1);
        % 帧局部区域
        frame = A(g(1,2):g(2,2),g(1,1):g(2,1),:);
        % 画图
        imagesc(frame(:,:,1));
        drawnow
        % 写入
        % frame = getframe(gcf);
        writeVideo(p,frame)
    end

    % 关闭视频
    close(p)

