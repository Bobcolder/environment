# Generated with SMOP  0.41
from smop.libsmop import *
# imcrop/matlab/FADE.m

    
@function
def FADE(I=None,*args,**kwargs):
    varargin = FADE.varargin
    nargin = FADE.nargin

    # Input: a test image, I
    # Output: perceptual fog density D and fog density map D_map
    # Detail explanation:
    # L. K. Choi, J. You, and A. C. Bovik, "Referenceless Prediction of Perceptual Fog Density and Perceptual Image Defogging",
    # IEEE Transactions on Image Processing, to appear (2015).
    
    ## Basic setup
    ps=8
# imcrop/matlab/FADE.m:9
    
    # Size of a test image for checking possilbe distinct patches
    row,col,dim=size(I,nargout=3)
# imcrop/matlab/FADE.m:11
    patch_row_num=floor(row / ps)
# imcrop/matlab/FADE.m:12
    patch_col_num=floor(col / ps)
# imcrop/matlab/FADE.m:13
    I=I[1:dot(patch_row_num,ps), 1:dot(patch_col_num,ps), :]  # : <- arange(1,3)
# imcrop/matlab/FADE.m:14
    row,col,dim=size(I,nargout=3)
# imcrop/matlab/FADE.m:15
    patch_row_num=floor(row / ps)
# imcrop/matlab/FADE.m:16
    patch_col_num=floor(col / ps)
# imcrop/matlab/FADE.m:17
    I=I[1:dot(patch_row_num,ps), 1:dot(patch_col_num,ps), :]  # 1: <- arange
# imcrop/matlab/FADE.m:18
    
    R=float(I(arange(),arange(),1))  # float <- double
# imcrop/matlab/FADE.m:20
    
    G=float(I(arange(),arange(),2))
# imcrop/matlab/FADE.m:21
    
    B=float(I(arange(),arange(),3))
# imcrop/matlab/FADE.m:22
    
    Ig=float(rgb2gray(I))
# imcrop/matlab/FADE.m:23
    
    # Dark channel prior image: Id, pixel-wise, scaled to [0 1]
    Irn=R / 255
# imcrop/matlab/FADE.m:25
    Ign=G / 255
# imcrop/matlab/FADE.m:26
    Ibn=B / 255
# imcrop/matlab/FADE.m:27
    Id=min(min(Irn,Ign),Ibn)
# imcrop/matlab/FADE.m:28
    
    I_hsv=rgb2hsv(I)
# imcrop/matlab/FADE.m:30
    Is=I_hsv(arange(),arange(),2)
# imcrop/matlab/FADE.m:31
    
    MSCN_window=fspecial('gaussian',7,7 / 6)
# imcrop/matlab/FADE.m:33
    MSCN_window=MSCN_window / sum(sum(MSCN_window))
# imcrop/matlab/FADE.m:34
    warning('off')
    mu=imfilter(Ig,MSCN_window,'replicate')
# imcrop/matlab/FADE.m:36
    mu_sq=multiply(mu,mu)
# imcrop/matlab/FADE.m:37
    sigma=sqrt(abs(imfilter(multiply(Ig,Ig),MSCN_window,'replicate') - mu_sq))
# imcrop/matlab/FADE.m:38
    MSCN=(Ig - mu) / (sigma + 1)
# imcrop/matlab/FADE.m:39
    cv=sigma / mu
# imcrop/matlab/FADE.m:40
    
    rg=R - G
# imcrop/matlab/FADE.m:42
    by=dot(0.5,(R + G)) - B
# imcrop/matlab/FADE.m:43
    
    # f1
    MSCN_var=reshape(nanvar(im2col(MSCN,concat([ps,ps]),'distinct')),concat([row,col]) / ps)
# imcrop/matlab/FADE.m:47
    
    MSCN_V_pair_col=im2col((multiply(MSCN,circshift(MSCN,concat([1,0])))),concat([ps,ps]),'distinct')
# imcrop/matlab/FADE.m:49
    
    MSCN_V_pair_col_temp1=copy(MSCN_V_pair_col)
# imcrop/matlab/FADE.m:50
    MSCN_V_pair_col_temp1[MSCN_V_pair_col_temp1 > 0]=NaN
# imcrop/matlab/FADE.m:50
    MSCN_V_pair_col_temp2=copy(MSCN_V_pair_col)
# imcrop/matlab/FADE.m:51
    MSCN_V_pair_col_temp2[MSCN_V_pair_col_temp2 < 0]=NaN
# imcrop/matlab/FADE.m:51
    MSCN_V_pair_L_var=reshape(nanvar(MSCN_V_pair_col_temp1),concat([row,col]) / ps)
# imcrop/matlab/FADE.m:52
    MSCN_V_pair_R_var=reshape(nanvar(MSCN_V_pair_col_temp2),concat([row,col]) / ps)
# imcrop/matlab/FADE.m:53
    
    Mean_sigma=reshape(mean(im2col(sigma,concat([ps,ps]),'distinct')),concat([row,col]) / ps)
# imcrop/matlab/FADE.m:55
    
    Mean_cv=reshape(mean(im2col(cv,concat([ps,ps]),'distinct')),concat([row,col]) / ps)
# imcrop/matlab/FADE.m:57
    
    CE_gray,CE_by,CE_rg=CE(I,nargout=3)
# imcrop/matlab/FADE.m:59
    Mean_CE_gray=reshape(mean(im2col(CE_gray,concat([ps,ps]),'distinct')),concat([row,col]) / ps)
# imcrop/matlab/FADE.m:60
    Mean_CE_by=reshape(mean(im2col(CE_by,concat([ps,ps]),'distinct')),concat([row,col]) / ps)
# imcrop/matlab/FADE.m:61
    Mean_CE_rg=reshape(mean(im2col(CE_rg,concat([ps,ps]),'distinct')),concat([row,col]) / ps)
# imcrop/matlab/FADE.m:62
    
    IE_temp=num2cell(im2col(uint8(Ig),concat([ps,ps]),'distinct'),1)
# imcrop/matlab/FADE.m:64
    IE=reshape(cellfun(entropy,IE_temp),concat([row,col]) / ps)
# imcrop/matlab/FADE.m:65
    
    Mean_Id=reshape(mean(im2col(Id,concat([ps,ps]),'distinct')),concat([row,col]) / ps)
# imcrop/matlab/FADE.m:67
    
    Mean_Is=reshape(mean(im2col(Is,concat([ps,ps]),'distinct')),concat([row,col]) / ps)
# imcrop/matlab/FADE.m:69
    
    CF=reshape(sqrt(std(im2col(rg,concat([ps,ps]),'distinct')) ** 2 + std(im2col(by,concat([ps,ps]),'distinct')) ** 2) + dot(0.3,sqrt(mean(im2col(rg,concat([ps,ps]),'distinct')) ** 2 + mean(im2col(by,concat([ps,ps]),'distinct')) ** 2)),concat([row,col]) / ps)
# imcrop/matlab/FADE.m:71
    feat=concat([ravel(MSCN_var),ravel(MSCN_V_pair_R_var),ravel(MSCN_V_pair_L_var),ravel(Mean_sigma),ravel(Mean_cv),ravel(Mean_CE_gray),ravel(Mean_CE_by),ravel(Mean_CE_rg),ravel(IE),ravel(Mean_Id),ravel(Mean_Is),ravel(CF)])
# imcrop/matlab/FADE.m:72
    feat=log(1 + feat)
# imcrop/matlab/FADE.m:73
    
    #Df (foggy level distance) for each patch
        # load natural fogfree image features (mu, cov)
    load('natural_fogfree_image_features_ps8.mat')
    
    mu_fog_param_patch=copy(feat)
# imcrop/matlab/FADE.m:80
    cov_fog_param_patch=nanvar(feat.T).T
# imcrop/matlab/FADE.m:81
    
    feature_size=size(feat,2)
# imcrop/matlab/FADE.m:83
    mu_matrix=repmat(mu_fogfreeparam,concat([size(feat,1),1])) - mu_fog_param_patch
# imcrop/matlab/FADE.m:84
    cov_temp1=[]
# imcrop/matlab/FADE.m:85
    cov_temp1[cumsum(multiply(feature_size,ones(1,length(cov_fog_param_patch))))]=1
# imcrop/matlab/FADE.m:86
    cov_temp2=cov_fog_param_patch(cumsum(cov_temp1) - cov_temp1 + 1,arange())
# imcrop/matlab/FADE.m:87
    cov_temp3=repmat(cov_temp2,concat([1,feature_size]))
# imcrop/matlab/FADE.m:88
    cov_temp4=repmat(cov_fogfreeparam,concat([length(cov_fog_param_patch),1]))
# imcrop/matlab/FADE.m:89
    cov_matrix=(cov_temp3 + cov_temp4) / 2
# imcrop/matlab/FADE.m:90
    
    mu_cell=num2cell(mu_matrix,2)
# imcrop/matlab/FADE.m:92
    cov_cell=mat2cell(cov_matrix,dot(feature_size,ones(1,size(mu_matrix,1))),feature_size)
# imcrop/matlab/FADE.m:93
    mu_transpose_cell=num2cell(mu_matrix.T,1)
# imcrop/matlab/FADE.m:94
    
    distance_patch=sqrt(cell2mat(cellfun(mtimes,cellfun(mrdivide,mu_cell,cov_cell,'UniformOutput',0),mu_transpose_cell.T,'UniformOutput',0)))
# imcrop/matlab/FADE.m:96
    Df=nanmean(distance_patch)
# imcrop/matlab/FADE.m:97
    
    Df_map=reshape(distance_patch,concat([row,col]) / ps)
# imcrop/matlab/FADE.m:98
    clear('mu_matrix','cov_matrix','mu_cell','cov_cell','mu_transpose_cell','distance_patch')
    #Dff
        # load natural foggy image features (mu, cov)
    load('natural_foggy_image_features_ps8.mat')
    
    mu_matrix=repmat(mu_foggyparam,concat([size(feat,1),1])) - mu_fog_param_patch
# imcrop/matlab/FADE.m:105
    cov_temp5=repmat(cov_foggyparam,concat([length(cov_fog_param_patch),1]))
# imcrop/matlab/FADE.m:106
    cov_matrix=(cov_temp3 + cov_temp5) / 2
# imcrop/matlab/FADE.m:107
    
    mu_cell=num2cell(mu_matrix,2)
# imcrop/matlab/FADE.m:109
    cov_cell=mat2cell(cov_matrix,dot(feature_size,ones(1,size(mu_matrix,1))),feature_size)
# imcrop/matlab/FADE.m:110
    mu_transpose_cell=num2cell(mu_matrix.T,1)
# imcrop/matlab/FADE.m:111
    
    distance_patch=sqrt(cell2mat(cellfun(mtimes,cellfun(mrdivide,mu_cell,cov_cell,'UniformOutput',0),mu_transpose_cell.T,'UniformOutput',0)))
# imcrop/matlab/FADE.m:113
    Dff=nanmean(distance_patch)
# imcrop/matlab/FADE.m:114
    
    Dff_map=reshape(distance_patch,concat([row,col]) / ps)
# imcrop/matlab/FADE.m:115
    clear('mu_matrix','cov_matrix','mu_cell','cov_cell','mu_transpose_cell')
    ## Perceptual fog density and density map
    D=Df / (Dff + 1)
# imcrop/matlab/FADE.m:118
    D_map=Df_map / (Dff_map + 1)
# imcrop/matlab/FADE.m:119
    return D,D_map
    
if __name__ == '__main__':
    pass
    