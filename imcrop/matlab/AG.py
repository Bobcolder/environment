# Generated with SMOP  0.41
from smop.libsmop import *
# AG.m

    
@function
def AG(inputImg=None,*args,**kwargs):
    varargin = AG.varargin
    nargin = AG.nargin

    image1=copy(inputImg)
# AG.m:2
#     image1=im2double(image1)
# AG.m:3
#     img=rgb2gray(image1)
    img = image1
# AG.m:4
    m,n=size(img,nargout=2)
# AG.m:5
    gradval=zeros(m,n)
# AG.m:7
    
    diffX=zeros(m,n)
# AG.m:8
    
    diffY=zeros(m,n)
# AG.m:9
    
    tempX=zeros(m,n)
# AG.m:10
    tempY=zeros(m,n)
# AG.m:11
    tempX[arange(1,m),arange(1,(n - 1))]=img[arange(1,m),arange(2,n)]  # [] for TypeError: 'matlabarray' object is not callable
# AG.m:12
    tempY[arange(1,(m - 1)),arange(1,n)]=img[arange(2,m),arange(1,n)]
# AG.m:13
    diffX=tempX - img
# AG.m:14
    diffY=tempY - img
# AG.m:15
    diffX[arange(1,m),n]=0
# AG.m:16
    
    diffY[m,arange(1,n)]=0
# AG.m:17
    diffX=multiply(diffX,diffX)
# AG.m:18
    diffY=multiply(diffY,diffY)
# AG.m:19
    AVEGRAD=(diffX + diffY) / 2
# AG.m:20
    AVEGRAD=sum(sum(sqrt(AVEGRAD)))
# AG.m:21
    AVEGRAD=AVEGRAD / (dot((m - 1),(n - 1)))
# AG.m:22
#     AVEGRAD=dot(AVEGRAD,255)
# AG.m:23
    return AVEGRAD
    
if __name__ == '__main__':
    pass
    