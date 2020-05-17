import torch.nn as nn
import torch
import torch.nn.functional as F 

#wrapper
#Convolution with batch noramlization
class BasicConv2d_bn(nn.Module):
    '''
    This class will be used to perform 2D Convolution with Batch Normalization
    in_channels:  input depth
    out_channels output depth
    padding(bool): if true padding of (kernal_size-1)//2
    '''
    def __init__(self,in_channels,out_channels,kernel_size,stride,pad=True):
        super(BasicConv2d_bn,self).__init__()

        if pad:
            #standard padding equation to maintain spatial dimension
            padding=(kernel_size-1)//2
        else:
            padding=0
        
        #convo
        self.Conv2d= nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride,padding=padding)
        #batch norm
        self.bn= nn.BatchNorm2d(out_channels, 0.001)

    def forward(self,x):
        return F.leaky_relu(self.bn(self.Conv2d(x)),inplace=True)




#wrapper
#Convolution without batch normalization
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride,pad=True):
        
        super(BasicConv2d,self).__init__()
        
        if pad:
            #standard padding equation to maintain spatial dimension
            padding=(kernel_size-1)//2 
        else:
            padding=0

        #convo
        self.Conv2d=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                kernel_size=kernel_size,stride=stride)
    
    def forward(self,x):
        return self.Conv2d(x)



#Darknet Model
class YoloV3Model(nn.Module):
    def __init__(self,num_classes=80):
        super(YoloV3Model,self).__init__()
        '''
        Output of convolution layer is passed through batch normalization
        Instead of dropout we use batch norm
        
        conv_name= nn.Conv2d(in,out,kenel_size,stride)
        batc_norm=nn.BatchNorm(out,0.001)
        
        in forward pass the output of convo layer will be passed through Batch Norm
        and the the out put will be squooshed by Leaky ReLU Activation function

        output=F.leank_relu(batch_norm(conv_name))
        '''
        self.Conv2d_1=BasicConv2d_bn(3,32,3,1) #args(in,out,kernel_size,stride,pad=True)
        self.Conv2d_2=BasicConv2d_bn(32,64,3,2)
        self.Conv2d_3=BasicConv2d_bn(64,32,1,1)
        self.Conv2d_4=BasicConv2d_bn(32,64,3,1)
        #shortcut
        self.Conv2d_5=BasicConv2d_bn(64,128,3,2)
        self.Conv2d_6=BasicConv2d_bn(128,64,1,1)
        self.Conv2d_7=BasicConv2d_bn(64,128,3,1)
        #shortcut
        self.Conv2d_8=BasicConv2d_bn(128,64,1,1)
        self.Conv2d_9=BasicConv2d_bn(64,128,3,1)
        #shortcut + #downsample route 1
        self.Conv2d_10=BasicConv2d_bn(128,256,3,2)
        self.Conv2d_11=BasicConv2d_bn(256,128,1,1)
        self.Conv2d_12=BasicConv2d_bn(128,256,3,1)
        #shortcut
        self.Conv2d_13=BasicConv2d_bn(256,128,1,1)
        self.Conv2d_14=BasicConv2d_bn(128,256,3,1)
        #shorcut
        self.Conv2d_15=BasicConv2d_bn(256,128,1,1)
        self.Conv2d_16=BasicConv2d_bn(128,256,3,1)
        #shorcut
        self.Conv2d_17=BasicConv2d_bn(256,128,1,1)
        self.Conv2d_18=BasicConv2d_bn(128,256,3,1)
        #shorcut
        self.Conv2d_19=BasicConv2d_bn(256,128,1,1)
        self.Conv2d_20=BasicConv2d_bn(128,256,3,1)
        #shortcut
        self.Conv2d_21=BasicConv2d_bn(256,128,1,1)
        self.Conv2d_22=BasicConv2d_bn(128,256,3,1)
        #shortcut
        self.Conv2d_23=BasicConv2d_bn(256,128,1,1)
        self.Conv2d_24=BasicConv2d_bn(128,256,3,1)
        #shortcut
        self.Conv2d_25=BasicConv2d_bn(256,128,1,1)
        self.Conv2d_26=BasicConv2d_bn(128,256,3,1)
        #shortcut + #downsample route 1
        self.Conv2d_27=BasicConv2d_bn(256,512,3,2)
        self.Conv2d_28=BasicConv2d_bn(512,256,1,1)
        self.Conv2d_29=BasicConv2d_bn(256,512,3,1)
        #shortcut
        self.Conv2d_30=BasicConv2d_bn(512,1024,1,1)
        self.Conv2d_31=BasicConv2d_bn(1024,512,3,1)
        #shortcut
        self.Conv2d_32=BasicConv2d_bn(512,1024,1,1)
        self.Conv2d_33=BasicConv2d_bn(1024,512,3,1)
        #shortcut
        self.Conv2d_34=BasicConv2d_bn(512,1024,1,1)
        self.Conv2d_35=BasicConv2d_bn(1024,512,3,1)
        #shortcut
        self.Conv2d_36=BasicConv2d_bn(512,256,1,1)
        self.Conv2d_37=BasicConv2d_bn(256,512,3,1)
        #shortcut
        self.Conv2d_38=BasicConv2d_bn(512,1024,1,1)
        self.Conv2d_39=BasicConv2d_bn(1024,512,3,1)
        #shortcut
        self.Conv2d_40=BasicConv2d_bn(512,1024,1,1)
        self.Conv2d_41=BasicConv2d_bn(1024,512,3,1)
        #shortcut
        self.Conv2d_42=BasicConv2d_bn(512,1024,1,1)
        self.Conv2d_43=BasicConv2d_bn(1024,512,3,1)
        #shortcut #downsample route 1
        self.Conv2d_44=BasicConv2d_bn(512,1024,3,2)
        self.Conv2d_45=BasicConv2d_bn(1024,512,1,1)
        self.Conv2d_46=BasicConv2d_bn(512,1024,3,1)
        #shortcut
        self.Conv2d_47=BasicConv2d_bn(1024,512,1,1)
        self.Conv2d_48=BasicConv2d_bn(512,1024,3,1)
        #shortcut
        self.Conv2d_49=BasicConv2d_bn(1024,512,1,1)
        self.Conv2d_50=BasicConv2d_bn(512,1024,3,1)
        #shortcut
        self.Conv2d_51=BasicConv2d_bn(1024,512,1,1)
        self.Conv2d_52=BasicConv2d_bn(512,1024,3,1)
        #shortcut
        self.Conv2d_53=BasicConv2d_bn(1024,512,1,1)
        self.Conv2d_54=BasicConv2d_bn(512,1024,3,1)
        self.Conv2d_55=BasicConv2d_bn(1024,512,1,1)
        self.Conv2d_56=BasicConv2d_bn(512,1024,3,1)
        self.Conv2d_57=BasicConv2d_bn(1024,512,1,1)#route->59
        self.Conv2d_58=BasicConv2d_bn(512,1024,3,1)#fully conncted layer is not required, so route previous layer after detection layer
        #Insert Detection Layer Here
        self.Conv2d_59=BasicConv2d(512,256,1,1) #note the out_channels of 59 are in_channels here, routing done.
        #upsample
        self.Conv2d_60=BasicConv2d_bn(768,256,1,1)
        self.Conv2d_61=BasicConv2d_bn(256,512,1,1)
        self.Conv2d_62 = BasicConv2d_bn(512, 256,1,1)
        self.Conv2d_63 = BasicConv2d_bn(256, 512,3,1) 
        self.Conv2d_64 = BasicConv2d_bn(512, 256,1,1) #route
        self.Conv2d_65 = BasicConv2d_bn(256, 512,3,1)
        #detction layer 2
        #route -1
        self.Conv2d_66 = BasicConv2d_bn(256, 128,1,1)
        #upsample
        #route
        self.Conv2d_67 = BasicConv2d_bn(384, 128,1,1)
        self.Conv2d_68 = BasicConv2d_bn(128, 256,3,1)
        self.Conv2d_69 = BasicConv2d_bn(256, 128,1,1)
        self.Conv2d_70 = BasicConv2d_bn(128, 256,3,1) 
        self.Conv2d_71 = BasicConv2d_bn(256, 128,1,1) 
        self.Conv2d_72 = BasicConv2d_bn(128, 256,3,1)
        #detection layer

        #upsample layer
        self.upsample=nn.Upsample(scale_factor=2)
        #calculating depth
        # No. of detection per grid * (5 + No. of classes)
        detection_depth=3*(5+num_classes)
        
        #prediction layers
        self.Conv2d_out3 = BasicConv2d(256, detection_depth,1,1)
        self.Conv2d_out2 = BasicConv2d(512, detection_depth,1,1)
        self.Conv2d_out1 = BasicConv2d(1024, detection_depth,1,1)
        
        

    def forward(self,x):    
        #3x416x416 x.shape
        x=self.Conv2d_1(x) 
        short=x=self.Conv2d_2(x)   #downsample  
        x=self.Conv2d_3(x)
        x=self.Conv2d_4(x)
        x=short+x   #short -3 #64x208x208

        short=x=self.Conv2d_5(x)    #downsample 
        x=self.Conv2d_6(x)
        x=self.Conv2d_7(x)
        short=x=short+x #short -3 #128x104x104

        x=self.Conv2d_8(x)
        x=self.Conv2d_9(x)
        x=x+short   #short -3

        short=x=self.Conv2d_10(x)   #downsample 
        x=self.Conv2d_11(x)
        x=self.Conv2d_12(x)
        short=x= short+x    #short -3 #256x52x52

        x=self.Conv2d_13(x)
        x=self.Conv2d_14(x)
        short=x=x+short     #short -3

        x=self.Conv2d_15(x)
        x=self.Conv2d_16(x)
        short=x=short+x     #short -3

        x=self.Conv2d_17(x)
        x=self.Conv2d_18(x)
        short=x=short+x   #short -3

        x=self.Conv2d_19(x)
        x=self.Conv2d_20(x)
        short=x=short+x   #short -3

        x=self.Conv2d_21(x)
        x=self.Conv2d_22(x)
        short=x=short+x   #short -3

        x=self.Conv2d_23(x)
        x=self.Conv2d_24(x)
        short=x=short+x #short -3

        x=self.Conv2d_25(x)
        x=self.Conv2d_26(x)
        route3=x=short+x   #short -3
        
        short=x=self.Conv2d_27(x)     #downsample 
        x=self.Conv2d_28(x)
        x=self.Conv2d_29(x)
        short=x=short+x   #short -3 #512x26x26

        x=self.Conv2d_30(x)
        x=self.Conv2d_31(x)
        short=x=short+x   #short -3

        x=self.Conv2d_32(x)
        x=self.Conv2d_33(x)
        short=x=short+x   #short -3

        x=self.Conv2d_34(x)
        x=self.Conv2d_35(x)
        short=x=short+x    #short -3

        x=self.Conv2d_36(x)
        x=self.Conv2d_37(x)
        short=x=short+x     #short -3

        x=self.Conv2d_38(x)
        x=self.Conv2d_39(x)
        short=x=short+x       #short -3

        x=self.Conv2d_40(x)
        x=self.Conv2d_41(x)
        short=x=short+x       #short -3

        x=self.Conv2d_42(x)
        x=self.Conv2d_43(x)
        route1=short=x=short+x       #short -3

        short=x=self.Conv2d_44(x)     #downsample
        x=self.Conv2d_45(x)
        x=self.Conv2d_46(x)
        short=x=short+x   #short -3   #1024x13x13

        x=self.Conv2d_47(x)
        x=self.Conv2d_48(x)
        short=x=short+x   #short -3

        x=self.Conv2d_49(x)
        x=self.Conv2d_50(x)
        short=x=short+x   #short -3

        x=self.Conv2d_51(x)
        x=self.Conv2d_52(x)
        short=x=short+x   #short -3

        x=self.Conv2d_53(x)
        x=self.Conv2d_54(x)
        short=x=short+x   #short -3

        x=self.Conv2d_55(x)
        x=self.Conv2d_56(x)

        route0=x=self.Conv2d_57(x)   #route0->detection1
        x=self.Conv2d_58(x)

        detection1=self.Conv2d_out1(x)  #512x13x13

        x=self.Conv2d_59(route0)
        x=self.upsample(x) #512x26x26
        x=torch.cat((x,route1),1) #concat along depth

        x = self.Conv2d_60(x)
        x = self.Conv2d_61(x)
        x = self.Conv2d_62(x)
        x = self.Conv2d_63(x)
        route2 = x = self.Conv2d_64(x)
        x = self.Conv2d_65(x)   
        
        detection2 = self.Conv2d_out2(x)  #255x26x26

        x = self.Conv2d_66(route2)           
        x = self.upsample(x)

        x = torch.cat((x, route3), 1)

        x = self.Conv2d_67(x)
        x = self.Conv2d_68(x)
        x = self.Conv2d_69(x)
        x = self.Conv2d_70(x)
        x = self.Conv2d_71(x)
        x = self.Conv2d_72(x)       #255x52x52

        detection3 = self.Conv2d_out3(x)

        return detection1, detection2, detection3



if __name__ == "__main__":
    import time
    since=time.time()

    model=YoloV3Model()
    x=torch.rand((1,3,416,416))

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,x=model.to(device),x.to(device)

    out1,out2,out3=model(x)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print("Done in : ", time.time()-since)



        