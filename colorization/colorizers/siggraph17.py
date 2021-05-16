import torch
import torch.nn as nn
import sklearn.neighbors as neighbors

from .base_color import *
from .util import *

# Has no backprop! Make sure to to call NeighborsEncLayer().zero_grad()
class NeighborsEncLayer(nn.Module):
    ''' Layer which encodes ab map into Q colors
    INPUTS    
        bottom[0]   Nx2xXxY     
    OUTPUTS
        top[0].data     NxQ     
    '''
    def __init__():
        super().__init__()

        self.neighbors = 10.
        self.sigma = 5.
        self.neighbors_enc = NeighborsEncode(self.neighbors, self.sigma, km_filepath='.pts_in_hull.npy')

        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]
        self.Q = self.neighbors_enc.K

    def forward():
        top[0].data[...] = self.nnenc.encode_points_mtx_nd(bottom[0].data[...],axis=1)
        # Do I actually move this here??
        top[0].reshape(self.N,self.Q,self.X,self.Y)

"""
class NeighborsEncLayer(caffe.Layer):
    ''' Layer which encodes ab map into Q colors
    INPUTS    
        bottom[0]   Nx2xXxY     
    OUTPUTS
        top[0].data     NxQ     
    '''
    def setup(self,bottom, top):
        warnings.filterwarnings("ignore")

        if len(bottom) == 0:
            raise Exception("Layer should have inputs")
        self.NN = 10.
        self.sigma = 5.
        self.ENC_DIR = './resources/'
        self.nnenc = NNEncode(self.NN,self.sigma,km_filepath=os.path.join(self.ENC_DIR,'pts_in_hull.npy'))

        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]
        self.Q = self.nnenc.K

    def reshape(self, bottom, top):
        top[0].reshape(self.N,self.Q,self.X,self.Y)
 
    def forward(self, bottom, top):
        top[0].data[...] = self.nnenc.encode_points_mtx_nd(bottom[0].data[...],axis=1)

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)
"""

class NeighborsEncode():
    ''' Encode points using NN search and Gaussian kernel '''
    def __init__(self,NN,sigma,km_filepath='',cc=-1):
        if(check_value(cc,-1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = neighbors.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc) # eventually replace with PyTorch nearest neighbors?

        self.alreadyUsed = False

    def encode_points_mtx_nd(self,pts_nd,axis=1,returnSparse=False,sameBlock=True):
        pts_flt = flatten_nd_array(pts_nd,axis=axis)
        P = pts_flt.shape[0]
        if(sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0 # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P,self.K))
            self.p_inds = np.arange(0,P,dtype='int')[:,na()]

        P = pts_flt.shape[0]

        (dists,inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists**2/(2*self.sigma**2))
        wts = wts/np.sum(wts,axis=1)[:,na()]

        self.pts_enc_flt[self.p_inds,inds] = wts
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt,pts_nd,axis=axis)

        return pts_enc_nd

    # I can't tell when either of these decode functions are ever called
    def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd,axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt,self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
        return pts_dec_nd

    def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
        pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
        pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
        if(returnEncode):
            return (pts_dec_nd,pts_1hot_nd)
        else:
            return pts_dec_nd

# Has no backprop! Make sure to to call PriorBoostLayer().zero_grad()
class PriorBoostLayer(nn.Module):
    ''' Layer boosts ab values based on their rarity
    INPUTS    
        bottom[0]       NxQxXxY     
    OUTPUTS
        top[0].data     Nx1xXxY
    '''
    def __init__():
        self.gamma = .5
        self.alpha = 1.
        self.pc = PriorFactor(self.alpha, gamma=self.gamma, priorFile='./prior_probs.npy')

        self.N = bottom[0].data.shape[0]
        self.Q = bottom[0].data.shape[1]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def forward():
        top[0].data[...] = self.pc.forward(bottom[0].data[...],axis=1)
        # I'm pretty sure I call this here?
        top[0].reshape(self.N,1,self.X,self.Y)

class PriorBoostLayer(caffe.Layer):
    ''' Layer boosts ab values based on their rarity
    INPUTS    
        bottom[0]       NxQxXxY     
    OUTPUTS
        top[0].data     Nx1xXxY
    '''
    def setup(self,bottom, top):
        if len(bottom) == 0:
            raise Exception("Layer should have inputs")

        self.ENC_DIR = './resources/'
        self.gamma = .5
        self.alpha = 1.
        self.pc = PriorFactor(self.alpha,gamma=self.gamma,priorFile=os.path.join(self.ENC_DIR,'prior_probs.npy'))

        self.N = bottom[0].data.shape[0]
        self.Q = bottom[0].data.shape[1]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
 
    def forward(self, bottom, top):
        top[0].data[...] = self.pc.forward(bottom[0].data[...],axis=1)

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class PriorFactor():
    ''' Class handles prior factor '''
    def __init__(self,alpha,gamma=0,verbose=True,priorFile=''):
        # INPUTS
        #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior**alpha
        #   gamma           integer     percentage to mix in uniform prior with empirical prior
        #   priorFile       file        file which contains prior probabilities across classes

        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = np.load(priorFile)

        # define uniform probability
        self.uni_probs = np.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs!=0] = 1.
        self.uni_probs = self.uni_probs/np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution       
        self.prior_mix = (1-self.gamma)*self.prior_probs + self.gamma*self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix**-self.alpha
        self.prior_factor = self.prior_factor/np.sum(self.prior_probs*self.prior_factor) # re-normalize

        # implied empirical prior
        self.implied_prior = self.prior_probs*self.prior_factor
        self.implied_prior = self.implied_prior/np.sum(self.implied_prior) # re-normalize

        if (self.verbose):
            self.print_correction_stats()

    def print_correction_stats(self):
        print 'Prior factor correction:'
        print '  (alpha,gamma) = (%.2f, %.2f)'%(self.alpha,self.gamma)
        print '  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)'%(np.min(self.prior_factor),np.max(self.prior_factor),np.mean(self.prior_factor),np.median(self.prior_factor),np.sum(self.prior_factor*self.prior_probs))

    def forward(self,data_ab_quant,axis=1):
        data_ab_maxind = np.argmax(data_ab_quant,axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if(axis==0):
            return corr_factor[na(),:]
        elif(axis==1):
            return corr_factor[:,na(),:]
        elif(axis==2):
            return corr_factor[:,:,na(),:]
        elif(axis==3):
            return corr_factor[:,:,:,na()]

class SIGGRAPHGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d, classes=529):
        super(SIGGRAPHGenerator, self).__init__()

        model_ss = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=8, groups=2)]

        # Conv1
        model1=[nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]
        # add a subsampling operation

        # Conv2
        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]
        # add a subsampling layer operation

        # Conv3
        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]
        # add a subsampling layer operation

        # Conv4
        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        # Conv5
        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        # Conv6
        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        # Conv7
        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        # Conv8
        model8up=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
        model3short8=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]

        model8=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[norm_layer(256),]

        # Conv9
        model9up=[nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),]
        model2short9=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        # add the two feature maps above        

        model9=[nn.ReLU(True),]
        model9+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model9+=[nn.ReLU(True),]
        model9+=[norm_layer(128),]

        # Conv10
        model10up=[nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True),]
        model1short10=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        # add the two feature maps above

        model10=[nn.ReLU(True),]
        model10+=[nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=True),]
        model10+=[nn.LeakyReLU(negative_slope=.2),]

        # Color encoding
        model_colorencode = [NeighborsEncLayer(),]

        # classification output
        model_class=[nn.Conv2d(256, classes, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),]

        # regression output
        model_out=[nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),]
        model_out+=[nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        self.model_colorencode = nn.Sequential(*model_colorencode)
        # is this the right place to call this?
        self.model_colorencode.zero_grad()

        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),])
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1),])

    def forward(self, input_A, input_B=None, mask_B=None):
        if(input_B is None):
            input_B = torch.cat((input_A*0, input_A*0), dim=1)
        if(mask_B is None):
            mask_B = input_A*0
        
        # if dev:
        #     input_B = input_B.to(dev)
        #     mask_B = mask_B.to(dev)
        normalized_A = self.normalize_l(input_A)
        normalized_B = self.normalize_ab(input_B)
        # print("normalized_A type", type(normalized_A))
        # print("normalized_B type", type(normalized_B))
        # print("normalized_A size", list(normalized_A.size()))
        # print("normalized_B size", list(normalized_B.size()))
        conv1_2 = self.model1(torch.cat((normalized_A,normalized_B,mask_B),dim=1))
        conv2_2 = self.model2(conv1_2[:,:,::2,::2])
        conv3_3 = self.model3(conv2_2[:,:,::2,::2])
        conv4_3 = self.model4(conv3_3[:,:,::2,::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)

        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)
        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)


        """
        out_reg = self.model_out(conv10_2)
        colorencode = self.model_colorencode(conv8_3)
        data_ab_ss = self.model_ab_ss(input_B)
        gt_ab_ss = self.model_colorencode(data_ab_ss)
        nongray_mask = self.model_nongraymask(data_ab_ss)
        prior_boost_nongray = prior_boost * nongray_mask

        conv8_boost = self.class_rebalance(conv8_3, prior_boost_nongray)
        # TODO: what
        class_reg = self.model_class(conv8_boost)

        return self.unnormalize_ab(class_reg)
        """

        out_class = self.model_class(conv8_3)

        conv9_up = self.model9up(conv8_3.detach()) + self.model2short9(conv2_2.detach())
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2.detach())
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        # TODO: Do you unnormalize this? Prob not
        # return self.unnormalize_ab(class_reg)
        return class_reg

def siggraph17(pretrained=True):
    model = SIGGRAPHGenerator()
    if(pretrained):
        import torch.utils.model_zoo as model_zoo
        model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth',map_location='cpu',check_hash=True))
    return model
