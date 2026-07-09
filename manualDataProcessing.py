import numpy as np




def manual_resize(pic:np.ndarray,uniform_size=(int,int)):
    pic_h,pic_w,pic_c=pic.shape
    uni_h,uni_w=uniform_size


    ratio_h=uni_h/pic_h
    ratio_w=uni_w/pic_w

    out=np.zeros((uni_h,uni_w,pic_c),dtype=np.float32)
    #输出

    for i in range(uni_h):
        pic_y=(i+0.5)*ratio_h-0.5
        #这个按比例放缩的坐标可能是浮点，因此要算加权平均
        y0=max(int(pic_y),0)
        y1=min(y0+1,pic_h-1)#防越界
        dy0=pic_y-y0
        dy1=y1-pic_y

        for j in range(uni_w):
            pic_x = (j + 0.5) * ratio_w - 0.5
            # 这个按比例放缩的坐标可能是浮点，因此要算加权平均
            x0 = max(int(pic_x), 0)
            x1 = min(x0 + 1, pic_w - 1)  # 防越界
            dx0 = pic_x - x0
            dx1 = x1 - pic_x


            f00=pic[y0,x0,:]
            f01=pic[y0,x1,:]
            f10=pic[y1,x0,:]
            f11=pic[y1,x1,:]

            w00 = dx1 * dy1
            w01 = dx0 * dy1
            w10 = dx1 * dy0
            w11 = dx0 * dy0

            out[i, j, :] = f00 * w00 + f01 * w01 + f10 * w10 + f11 * w11



        out=np.clip(out,0,255).astype(np.uint8)
    return out


def manual_processing(pic:np.ndarray,uniform_size=(int,int),
                      mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)):
    resized_pic=manual_resize(pic,uniform_size)

    #读取的本来就是rgb，不用改

    resized_pic_float=resized_pic/255.0

    out=np.zeros_like(resized_pic_float)
    for i in range(3):
        out[:,:,i]=(resized_pic_float[:,:,i]-mean[i])/std[i]



    out = out.transpose(2, 0, 1)
    return out








