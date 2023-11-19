import torch
import torch.nn as nn


class MultiScaleConv(nn.Module):
    """
    Multi-scale convolution block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1,1)),
                                  nn.ZeroPad2d([1, 1, 2, 0]),
                                  nn.Conv2d(out_channels, out_channels, (3,3), groups=out_channels),
                                  nn.Conv2d(out_channels, out_channels, (1,1)))
        self.skip = nn.Conv2d(in_channels, out_channels, (1,1))
        
    def forward(self, x):
        """ x: (B,C,T,F) """
        y = self.conv(x)
        y = y * self.skip(x)
        return y
    

class MultiOrderConv(nn.Module):
    """
    Multi-order convolution block
    """
    def __init__(self, C=64, order=5):
        super().__init__()
        self.C = C
        self.order = order
        
        self.pwconv1 = nn.Conv2d(C//2, 2*C, 1)
        self.pwconvN = nn.Conv2d(C, 1, 1)
        
        self.dwconvs = nn.ModuleList([])
        self.pwconvs = nn.ModuleList([])
        
        for i in range(order-1):
            n = order-1 - i
            pw_channels = int(C/2**n)
            dw_channels = int(2*C - C/2**n)
            # print(dw_channels, pw_channels)
            
            self.pwconvs.append(nn.Conv2d(pw_channels, pw_channels*2, 1))
            self.dwconvs.append(nn.Sequential(
                nn.ZeroPad2d([3, 3, 6, 0]),
                nn.Conv2d(dw_channels, dw_channels, (7,7), groups=dw_channels)))
    
    def forward(self, x):
        """ x: (B,C/2,T,F) -> (B,C,T,F) """
        assert(x.shape[1] == self.C//2)
        
        x = self.pwconv1(x)  # (B,2C,T,F)
        dw_x_in = x[:, :-int(self.C/2**(self.order-1))]
        pw_x_in = x[:, -int(self.C/2**(self.order-1)):]
        
        for i in range(self.order-1):     
            n = self.order-1 - i
                   
            dw_temp = self.dwconvs[i](dw_x_in)
            dw_x_out = dw_temp[:, :-int(self.C/2**n)]
            
            pw_temp = dw_temp[:, -int(self.C/2**n):]
            pw_x_out = self.pwconvs[i](pw_x_in * pw_temp)
            
            dw_x_in = dw_x_out
            pw_x_in = pw_x_out
            # print(dw_x_in.shape, pw_x_in.shape)   

        y = self.pwconvN(dw_x_in * pw_x_in)  # (B,C,T,F)
        
        return y
             

class LiteRTSE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lstm1 = nn.LSTM(257, 257, 1, batch_first=True)
        self.fc1 = nn.Linear(257, 257)
        
        self.msmo_conv = nn.Sequential(MultiScaleConv(1, 32),
                                       MultiScaleConv(32, 32),
                                       MultiOrderConv(64, 5))
        
        self.fc2 = nn.Linear(771, 257)
        self.lstm2 = nn.LSTM(257, 257, 1, batch_first=True)
        
        self.to_mag = nn.Linear(257, 257)
        self.to_real = nn.Linear(257, 257)
        self.to_imag = nn.Linear(257, 257)

    def forward(self, X):
        """ X: (B,F,T,2), spectrogram of noisy speech. """
        X = X.permute(0,3,2,1)  # (B,2,T,F)
        X_mag = (X[:,0]**2 + X[:,1]**2).pow(0.5)  # (B,T,F)
        X_pha = torch.atan2(X[:,1], X[:,0])       # (B,T,F)
        
        ### LSTM1
        M = torch.sigmoid(self.fc1(self.lstm1(X_mag)[0]))  # (B,T,F)
        Y1 = X_mag * M  # (B,T,F)
        
        ### Multi-Scale and Multi-Order Conv
        Y2 = self.msmo_conv(X_mag[:, None])[:,0]  # (B,C,T,F)
        
        ### LSTM2
        Y = torch.concat([X_mag, Y1, Y2], dim=-1)  # (B,T,3F)
        Z = self.fc2(Y)  # (B,T,F)
        Z = self.lstm2(Z)[0]  # (B,T,F)
        
        ### Residual Compensation
        R_mag = self.to_mag(Z)
        R_real = self.to_real(Z)
        R_imag = self.to_imag(Z)
        R_pha = torch.atan2(R_imag, R_real)
        
        S_mag = R_mag * (X_mag - Y1) + Y1
        S_pha_cos = torch.cos(X_pha) * torch.cos(R_pha) - torch.sin(X_pha) * torch.sin(R_pha)
        S_pha_sin = torch.sin(X_pha) * torch.cos(R_pha) + torch.cos(X_pha) * torch.sin(R_pha)
        
        S = torch.stack([S_mag * S_pha_cos, S_mag * S_pha_sin], dim=-1).permute(0,2,1,3)  # (B,F,T,2)
        
        return S


if __name__ == "__main__":
    model = LiteRTSE()
    
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (257, 63, 2), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print(flops, params)

    """causality check"""
    a = torch.randn(1, 257, 100, 2)
    b = torch.randn(1, 257, 100, 2)
    c = torch.randn(1, 257, 100, 2)
    x1 = torch.cat([a, b], dim=2)
    x2 = torch.cat([a, c], dim=2)
    y1 = model(x1)
    y2 = model(x2)
    print((y1[:,:,:100,:] - y2[:,:,:100,:]).abs().max())
    print((y1[:,:,100:,:] - y2[:,:,100:,:]).abs().max())