#  Goodness of Fit Framework for Neural Population Models using  Time-rescaling Theorem

#####  Long Tao, Karoline E. Weber, Kensuke Arai, Uri T. Eden
[A common goodness-of-fit framework for neural population models using marked point process time-rescaling](https://www.biorxiv.org/content/early/2018/02/14/265850) (2018)

Source code to create Fig. 1, 2, 4 is [here](matlab_fig124)

##  popTRT tool
popTRT rescales spike times for marked spikes modeled with a marked point process model.  By default, we provide  the kernel based model which can be calculated from the spikes alone, or the user may provide parameters for a mixture of Gaussians model for the conditional intensity function.
###  Required packages
numpy
matplotlib
cython

### Installing popTRT
python setup.py install

### Using popTRT
dataformat

####  Marked spike data
posx(t)  spk01(t) mark1...K

2 + K columns for 1-dimensional position, 0 or 1 spks in time bin and K dim mark (if spike == 1)

#### Directories
[popTRT](popTRT)    python and cython files 
[run](run)   the python run scripts where one would enter parameters of the run, ie model type, input data file location etc.
