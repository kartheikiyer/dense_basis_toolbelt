from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.interpolate import NearestNDInterpolator

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from ranger import Ranger

from tqdm import tqdm

from joblib import dump, load
import scipy.io as sio

import dense_basis as db

#-------------------------------------------------------------------------------
#--------------------------------Training Sets----------------------------------
#-------------------------------------------------------------------------------

def generate_training_set(priors, set_size = 100, random_seed = 42, fname = 'train_data'):

    rand_sfh_tuple, rand_Z, rand_Av, rand_z = priors.sample_all_params_safesSFR(random_seed = random_seed)
    rand_sfh, rand_time = db.tuple_to_sfh(rand_sfh_tuple, zval = rand_z)
    rand_spec, rand_lam = db.make_spec(rand_sfh_tuple, rand_Z, rand_Av, rand_z, return_lam = True)
    spec_truths = (rand_sfh_tuple[0], rand_sfh_tuple[1], rand_sfh_tuple[3:], rand_Z, rand_Av, rand_z)

    train_specs = np.zeros((len(rand_spec), set_size))
    train_lam = rand_lam
    train_params = np.zeros((len(spec_truths), set_size))

    for i in tqdm(range(set_size)):
        rand_sfh_tuple, rand_Z, rand_Av, rand_z = priors.sample_all_params_safesSFR(random_seed = random_seed + i)
        rand_sfh, rand_time = db.tuple_to_sfh(rand_sfh_tuple, zval = rand_z)
        rand_spec, rand_lam = db.make_spec(rand_sfh_tuple, rand_Z, rand_Av, rand_z, return_lam = True)
        _, sfr_true, mstar_true = db.make_spec(rand_sfh_tuple, rand_Z, rand_Av, rand_z, return_ms = True)
        train_params[0:,i] = np.hstack((mstar_true, sfr_true, rand_sfh_tuple[3:], rand_Z, rand_Av, rand_z))
        train_specs[0:,i] = rand_spec

    sio.savemat('training_sets/'+fname+'_N_'+str(set_size)+'.mat',mdict={'train_specs':train_specs, 'train_params':train_params, 'train_lam': train_lam})

    return

def load_training_set(fname, set_size):

    cat = sio.loadmat('training_sets/'+fname+'_N_'+str(set_size)+'.mat')
    train_specs = cat['train_specs']
    train_params = cat['train_params']
    train_lam = cat['train_lam'].ravel()

    return train_specs, train_params, train_lam

#-------------------------------------------------------------------------------
#--------------------------Pytorch NN backend-----------------------------------
#-------------------------------------------------------------------------------

def mish(x):
    return (x*torch.tanh(F.softplus(x)))

class Net_mish_ranger(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net_mish_ranger, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.hidden3 = torch.nn.Linear(n_hidden, n_output)   # hidden layer
        self.predict = torch.nn.Linear(n_output, n_output)   # output layer

    def forward(self, x):
        x = mish(self.hidden1(x))      # activation function for hidden layer
        x = mish(self.hidden2(x))
        x = mish(self.hidden3(x))
        x = self.predict(x)             # linear output
        return x

#-------------------------Training the PyTorch NN-------------------------------

def generate_PCA_NN_setup(train_params, train_specs, n_pca = 20, learningRate = 0.01, epochs = 10000, opt = 'Ranger', savefig = 'False'):

    if train_specs.shape[1] < 1000:
        print('check the shape of train_specs, either transpose the array or too few samples.')

    # fit the PCA and get coefficients
    pca = PCA(n_components = n_pca)
    pca.fit(np.log10(train_specs.T))
    train_spec_pca = pca.transform(np.log10(train_specs.T))

    # split the dataset into train and test

    y_val = (train_spec_pca)
    x_data= train_params.T
    x_data[x_data<-10] = -10 # to account for -inf values in log SFR
    X_train, X_eval, y_train, y_eval = train_test_split(x_data,y_val,test_size=0.1,random_state=42)

    inputDim = X_train.shape[1]       # takes variable 'x'
    outputDim = y_train.shape[1]       # takes variable 'y'

    X_validate = Variable(torch.from_numpy(X_eval)).float()
    y_validate = Variable(torch.from_numpy(y_eval)).float()

    # initialize the network and train it

    model = Net_mish_ranger(inputDim, 10, outputDim)

    criterion = torch.nn.MSELoss()
    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    elif opt == 'Ranger':
        optimizer = Ranger(model.parameters(), lr=learningRate)
    elif opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    else:
        raise Exception('Unknown optimizer: choose from [Adam, Ranger]')

    lossvals = np.zeros((epochs,))
    test_lossvals = np.zeros((epochs,))

    for epoch in tqdm(range(epochs)):

        # Converting inputs and labels to Variable

        inputs = Variable(torch.from_numpy(X_train)).float()
        labels = Variable(torch.from_numpy(y_train)).float()

        # Clear gradient buffers because we don't want any gradient from
        # previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        #print(loss)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        #print('epoch {}, loss {}'.format(epoch, loss.item()))
        lossvals[epoch] = loss.item()
        test_lossvals[epoch] = criterion(model(X_validate), y_validate).item()

        if savefig == True:
            galids = np.arange(10)
            if np.remainder(epoch,100) == 0:
                for galid in galids:
                    test_theta = train_params[0:,galid]
                    test_spec = train_specs[0:,galid]
                    pred_spec = predict_NN_spec(test_theta, model, pca)

                    plt.figure(figsize=(12,6))
                    plt.plot(rand_lam, test_spec,'k',label='truth')
                    plt.plot(rand_lam, pred_spec,label='NN')
                    plt.xscale('log');plt.xlim(1e3,1e5)
                    plt.ylim(0,np.amax(test_spec[rand_lam<1e5])*1.2)
                    plt.yscale('log');plt.ylim(1e-2,np.amax(test_spec[rand_lam<1e5])*1.2)
                    plt.xlabel('$\lambda$ [$\AA$]');plt.ylabel(r'F$\nu$ [$\mu$Jy]');plt.legend(edgecolor='w')
                    plt.savefig('training_figs/SED_NN_training_galid_'+str(galid)+'_epoch_'+str(epoch+epochs)+'.png',bbox_inches='tight')
                    plt.close()

    return model, pca, lossvals, test_lossvals
