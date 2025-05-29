clear
m=mobiledev

cam =camera(m,'back')

img=snapshot (cam,'manual')

img=imresize(img,[244,244,1]):
image (img)

% load net
load googlenet

nnet = googlenet

label=classify(nnet,img)

