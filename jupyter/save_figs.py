import pickle
import matplotlib.pyplot as plt
import tqdm
from KDEpy import FFTKDE
import numpy as np

if __name__=='__main__':
  entropy_all=pickle.load(open('/usr/commondata/weather/code/GraghPool/jupyter/entropy_all.pkl', 'rb'))
  for dataname in ['DD','PROTEINS','NCI1','NCI109','Mutagenicity','ENZYMES']:
    for k in tqdm.tqdm(range(10)):
        entropy=entropy_all['ENZYMES'][k]
        for channel in range(entropy.shape[1]):
            value_support=entropy[:,channel][entropy[:,channel]>=0]
            x_anchor=np.linspace(value_support.min(),value_support.max(),1000)
            P = FFTKDE(bw="silverman").fit(value_support).evaluate(x_anchor)
            plt.scatter(x_anchor,P)
            plt.savefig('/usr/commondata/weather/code/GraghPool/jupyter/figs/{}_{}_{}.png'.format(dataname,channel,k+1))
            plt.close()